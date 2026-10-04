"""Downloader regression tests: no real network calls or benchmark downloads."""

import io
from unittest.mock import MagicMock
from zipfile import BadZipFile, ZipFile

import pytest
import requests
from trustllm import dataset_download


def archive_bytes(files):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def http(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    get = MagicMock(return_value=response)
    monkeypatch.setattr(dataset_download.requests, "get", get)
    return get, response


def test_default_destination(tmp_path, monkeypatch, http):
    monkeypatch.chdir(tmp_path)
    get, response = http
    payload = archive_bytes({"dataset/safety/example.json": '[{"prompt": "hello"}]'})
    response.iter_content.return_value = [payload[:20], b"", payload[20:]]
    assert dataset_download.download_dataset() is None
    assert (tmp_path / "data/dataset/safety/example.json").read_text() == '[{"prompt": "hello"}]'
    assert not (tmp_path / "data/dataset.zip").exists()
    get.assert_called_once_with(dataset_download.DATASET_URL, stream=True, timeout=(10, 120))


def test_explicit_path_overwrites_dataset_file(tmp_path, http):
    destination = tmp_path / "custom"
    destination.mkdir()
    (destination / "existing.txt").write_text("old")
    http[1].iter_content.return_value = [archive_bytes({"existing.txt": "new"})]
    dataset_download.download_dataset(destination)
    assert (destination / "existing.txt").read_text() == "new"


def test_http_failure_does_not_touch_existing_files(tmp_path, http):
    (tmp_path / "existing.txt").write_text("keep")
    http[1].raise_for_status.side_effect = requests.HTTPError("404")
    with pytest.raises(requests.HTTPError):
        dataset_download.download_dataset(tmp_path)
    assert (tmp_path / "existing.txt").read_text() == "keep"
    http[1].iter_content.assert_not_called()


def test_interrupted_download_writes_no_dataset(tmp_path, http):
    def chunks(**kwargs):
        yield b"partial archive"
        raise requests.ConnectionError("interrupted")

    http[1].iter_content.side_effect = chunks
    destination = tmp_path / "uncreated"
    with pytest.raises(requests.ConnectionError):
        dataset_download.download_dataset(destination)
    assert not destination.exists()


def test_invalid_archive(tmp_path, http):
    http[1].iter_content.return_value = [b"not a zip"]
    with pytest.raises(BadZipFile):
        dataset_download.download_dataset(tmp_path / "uncreated")
    assert not (tmp_path / "uncreated").exists()


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "dataset/../../outside.txt"])
def test_rejects_archive_escape_before_extracting_any_file(tmp_path, http, name):
    http[1].iter_content.return_value = [archive_bytes({"valid.txt": "ok", name: "bad"})]
    destination = tmp_path / "download"
    with pytest.raises(ValueError, match="Unsafe archive path"):
        dataset_download.download_dataset(destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_rejects_existing_symlink_escape(tmp_path, http):
    destination = tmp_path / "download"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "dataset").symlink_to(outside, target_is_directory=True)
    http[1].iter_content.return_value = [archive_bytes({"dataset/file.txt": "bad"})]
    with pytest.raises(ValueError, match="Unsafe archive path"):
        dataset_download.download_dataset(destination)
    assert not (outside / "file.txt").exists()
