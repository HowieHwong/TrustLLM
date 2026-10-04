"""Real CPU inference on a tiny locally constructed model; no model downloads."""

import json

import pytest


@pytest.mark.parametrize("chat_template", [False, True])
def test_real_local_checkpoint_generation_and_resume(tmp_path, chat_template):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trustllm import generate

    torch.manual_seed(1)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "reply": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]"
    )
    if chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
    model_dir = tmp_path / "tiny-model"
    tokenizer.save_pretrained(model_dir)
    config = GPT2Config(
        vocab_size=4,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    # Force a known non-special continuation while exercising actual forward/generate.
    with torch.no_grad():
        model.transformer.ln_f.weight.zero_()
        model.transformer.ln_f.bias.fill_(1)
        model.lm_head.weight[3].fill_(1)
    model.save_pretrained(model_dir)
    source = tmp_path / "tiny.json"
    source.write_text('[{"prompt":"hello world","label":"kept"}]')
    output = tmp_path / "output"
    result = generate(
        str(model_dir),
        backend="local",
        task="safety",
        data_path=source,
        output_dir=output,
        device="cpu",
        max_new_tokens=2,
        temperature=0,
    )
    assert result["successful"] == 1
    rows = json.loads((output / "tiny.json").read_text())
    assert rows[0]["res"] == "reply reply"
    assert rows[0]["label"] == "kept"
    resumed = generate(
        str(model_dir),
        backend="local",
        task="safety",
        data_path=source,
        output_dir=output,
        device="cpu",
        max_new_tokens=2,
        temperature=0,
        resume=True,
    )
    assert resumed["successful"] == 1
    assert resumed["resolved_model"]["device"] == "cpu"
