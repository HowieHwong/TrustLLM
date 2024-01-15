import os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from fastchat.model import load_model, get_conversation_template, add_model_args
import logging
import argparse
from trustllm.utils.generation_utils import *
from dotenv import load_dotenv
import os
import json
import threading
from tqdm import tqdm
import urllib3
import traceback


load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
model_mapping, online_model = get_models()



def generation_hf(prompt, tokenizer, model, temperature):
    """
    Generates a response using a Hugging Face model.

    :param prompt: The input text prompt for the model.
    :param tokenizer: The tokenizer associated with the model.
    :param model: The Hugging Face model used for text generation.
    :param temperature: The temperature setting for text generation.
    :return: The generated text as a string.
    """
    msg = prompt
    conv = get_conversation_template(args.model_path)
    conv.set_system_message('')
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(args.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs


def generation(model_name, prompt, tokenizer, model, temperature=None):
    """
    Generates a response using either an online or a local model.

    :param model_name: The name of the model.
    :param prompt: The input text prompt for the model.
    :param tokenizer: The tokenizer for the model.
    :param model: The model used for text generation.
    :param temperature: The temperature setting for text generation. Default is None.
    :return: The generated text as a string.
    """

    try:
        if model_name in online_model:
            ans = gen_online(model_name, prompt, temperature)
        else:
            ans = generation_hf(prompt, tokenizer, model, temperature)
        if not ans:
            raise ValueError("The response is NULL or an empty string!")
        return ans
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)


def process_element(el, model, model_name, tokenizer, index, temperature, key_name='prompt'):
    """
    Processes a single element (data point) using the specified model.

    :param el: A dictionary containing the data to be processed.
    :param model: The model to use for processing.
    :param model_name: The name of the model.
    :param tokenizer: The tokenizer for the model.
    :param index: The index of the element in the dataset.
    :param temperature: The temperature setting for generation.
    :param key_name: The key in the dictionary where the prompt is located.
    """
    
    try:
        # If 'res' key doesn't exist or its value is empty, generate a new response
        if "res" not in el or not el['res']:
            res = generation(model_name=model_name, prompt=el[key_name], tokenizer=tokenizer, model=model,
                             temperature=temperature)
            el['res'] = res  
    except Exception as e:
        # Print error message if there's an issue during processing
        print(f"Error processing element at index {index}: {e}")


def process_file(data_path, save_path, model_name, tokenizer, model, file_config, key_name='prompt'):
    """
    Processes a file containing multiple data points for text generation.

    :param data_path: Path to the input data file.
    :param save_path: Path where the processed data will be saved.
    :param model_name: The name of the model used for processing.
    :param tokenizer: The tokenizer for the model.
    :param model: The model to use for processing.
    :param file_config: Configuration settings for file processing.
    :param key_name: The key in the dictionary where the prompt is located.
    """
    if os.path.basename(data_path) not in file_config:
        return

    with open(data_path) as f:
        original_data = json.load(f)

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
    else:
        saved_data = original_data

    GROUP_SIZE = 8 if model_name in online_model else 1
    for i in tqdm(range(0, len(saved_data), GROUP_SIZE), desc=f"Processing {data_path}", leave=False):
        group_data = saved_data[i:i + GROUP_SIZE]
        threads = []

        for idx, el in enumerate(group_data):
            temperature = file_config.get(os.path.basename(data_path), 0.0)
            t = threading.Thread(target=process_element,
                                 args=(el, model, model_name, tokenizer, idx, temperature, key_name))
            t.start()
            threads.append(t)
        file_process.save_json(saved_data, f"{save_path}")

        # Wait for all threads to complete
        for t in threads:
            t.join()
    file_process.save_json(saved_data, f"{save_path}")


def run_task(model_name, model, tokenizer, base_dir, file_config, key_name='prompt'):
    """
    Runs a specific evaluation task based on provided parameters.

    :param model_name: The name of the model.
    :param model: The model used for processing.
    :param tokenizer: The tokenizer for the model.
    :param base_dir: Base directory containing test data files.
    :param file_config: Configuration settings for file processing.
    :param key_name: The key in the dictionary where the prompt is located.
    """

    test_data_dir = os.path.join(base_dir, 'test_data')
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    test_res_dir = os.path.join(base_dir, 'test_res', model_name)
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)

    file_list = os.listdir(os.path.join(base_dir, 'test_data'))
    for file in tqdm(file_list, desc="Processing files"):
        data_path = os.path.join(base_dir, 'test_data', file)
        save_path = os.path.join(base_dir, 'test_res', model_name, file)
        process_file(data_path, save_path, model_name, tokenizer, model, file_config, key_name)


def run_ethics(model_name, model, tokenizer):
    base_dir = os.path.join(args.data_path, 'ethics')
    file_config = {
        "emotional_awareness.json": 0.0,
        'explicit_moralchoice.json': 1.0,
        "implicit_ETHICS.json": 0.0,
        "implicit_SocialChemistry101.json": 0.0
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)


def run_privacy(model_name, model, tokenizer):
    base_dir = os.path.join(args.data_path, 'privacy')
    file_config = {
        'privacy_awareness_confAIde.json': 0.0,
        'privacy_awareness_query.json': 1.0,
        'privacy_leakage.json': 1.0,
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)


def run_fairness(model_name, model, tokenizer):
    base_dir = os.path.join(args.data_path, 'fairness')
    file_config = {
        "disparagement.json": 1.0,
        "preference.json": 1.0,
        "stereotype_agreement.json": 1.0,
        'stereotype_query_test.json': 1.0,
        'stereotype_recognition.json': 0.0,
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)


def run_truthfulness(model_name, model, tokenizer):
    base_dir = os.path.join(args.data_path, 'truthfulness')
    file_config = {
        'external.json': 0.0,
        'hallucination.json': 0.0,
        "golden_advfactuality.json": 1.0,
        "internal.json": 1.0,
        "sycophancy.json": 1.0
    }
    run_task(model_name, model, tokenizer, base_dir, file_config, )


def run_robustness(model_name, model, tokenizer):
    base_dir = os.path.join(args.data_path, 'robustness')
    file_config = {
        'ood_detection.json': 1.0,
        'ood_generalization.json': 0.0,
        'AdvGLUE.json': 0.0,
        'AdvInstruction.json': 1.0,
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)


def run_safety(model_name, tokenizer, model):
    base_dir = os.path.join(args.data_path, 'safety')
    file_config = {
        'jailbreak.json': 1.0,
        "exaggerated_safety.json": 1.0,
        'misuse.json': 1.0,

    }
    run_task(model_name, model, tokenizer, base_dir, file_config, )

def run_single_test(args):
    """
    Executes a single test based on specified parameters.

    :param args: Contains parameters like test type, model name, and other configurations.
    :return: "OK" if successful, None otherwise.
    """
    model_name = args.model_name
    print(f"Beginning generation with {args.test_type} evaluation at temperature {args.temperature}.")
    print(f"Evaluation target model: {model_name}")

    model, tokenizer = (None, None) if model_name in online_model else load_model(
        args.model_path,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    test_functions = {
        'robustness': run_robustness,
        'truthfulness': run_truthfulness,
        'fairness': run_fairness,
        'ethics': run_ethics,
        'safety': run_safety,
        'privacy': run_privacy
    }

    test_func = test_functions.get(args.test_type)
    if test_func:
        test_func(model_name=model_name, model=model, tokenizer=tokenizer,)
        return "OK"
    else:
        print("Invalid test_type. Please provide a valid test_type.")
        return None


@torch.inference_mode()
def main(args, max_retries=10, retry_interval=3):
    """
    Main function to orchestrate the test runs with retries.

    :param args: Command-line arguments for the test run.
    :param max_retries: Maximum attempts to run the test.
    :param retry_interval: Time interval between retries in seconds.
    :return: Final state of the test run.
    """
    if not os.path.exists(args.data_path):
        print(f"Dataset path {args.data_path} does not exist.")
        return None

    args.model_name = model_mapping.get(args.model_path, "")

    for attempt in range(max_retries):
        try:
            state = run_single_test(args)
            if state:
                print(f"Test function successful on attempt {attempt + 1}")
                return state
        except Exception as e:
            print(f"Test function failed on attempt {attempt + 1}: {e}")
            print(f"Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    print("Test failed after maximum retries.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--test_type", type=str, default='')
    parser.add_argument("--data_path", type=str, default='TrustLLM')
    args = parser.parse_args()
    state = main(args, )
    print(state)