import os,time
import torch


from fastchat.model import load_model, get_conversation_template, add_model_args
import logging
import argparse
import json
from tqdm import tqdm
from utils import *
from dotenv import load_dotenv
import openai
import traceback
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import random
from tqdm import tqdm

import os
import json
import threading
import queue
from tqdm import tqdm

load_dotenv()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import traceback


model_mapping,online_model=get_models()


GROUP_SIZE = 10

def generation_hf(prompt,tokenizer,model,temperature):
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
        do_sample=True if temperature> 1e-5 else False,
        temperature=temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs


def generation(model_name, prompt, tokenizer, model, temperature=None):

    if temperature is None:
        temperature=args.temperature
    print(temperature)
    
    try:
        if model_name in online_model:
            ans=gen_online(model_name, prompt, temperature)   
        else:
            ans = generation_hf(prompt, tokenizer,model,temperature)
        if not ans:
            raise ValueError("The response is NULL or an empty string!")
        return ans
    except Exception as e:
        tb = traceback.format_exc()  # 获取完整的错误堆栈信息

        print(tb)  


def process_element(el, model,model_name, tokenizer, index, temperature,key_name='prompt' ):
    try:
        if "res" not in el.keys():
            res = generation(model_name=model_name, prompt=el[key_name], tokenizer=tokenizer, model=model, temperature=temperature)
            el['res'] = res
            
        elif not el.get('res'):
            res = generation(model_name=model_name, prompt=el[key_name], tokenizer=tokenizer, model=model, temperature=temperature)
            el['res'] = res


    except Exception as e:
        print(f"Error processing element at index {index}: {e}")


def process_file(data_path, save_path, model_name, tokenizer, model, file_config, key_name='prompt' ):
    
   
    GROUP_SIZE = 8 if model_name in online_model else 1
    #GROUP_SIZE = 1
    if os.path.basename(data_path) not in file_config:
        return
    
    with open(data_path) as f:
        original_data = json.load(f)
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
    else:
        saved_data=original_data

    #start_idx = len(saved_data)
    #remaining_data = original_data[start_idx:]

    # result_queue = queue.Queue()
    print(len(saved_data))
    
    for i in tqdm(range(0, len(saved_data), GROUP_SIZE), desc=f"Processing {data_path}", leave=False):
        group_data = saved_data[i:i + GROUP_SIZE]
        threads = []

        for idx, el in enumerate(group_data):
            temperature = file_config.get(os.path.basename(data_path), 0.0)
            t = threading.Thread(target=process_element, args=(el, model,model_name, tokenizer, idx, temperature,key_name))
            t.start()
            threads.append(t)

        save_path2=save_path.replace(".json","111.json")
        save_json(saved_data, f"{save_path}")

        # Wait for all threads to complete
        for t in threads:
            t.join()
    save_path2=save_path.replace(".json","111.json")
    save_json(saved_data, f"{save_path}")
def run_task(model_name, model, tokenizer, base_dir, file_config,key_name='prompt' ):

    assert os.path.exists(os.path.join(base_dir, 'test_data'))
    assert os.path.exists(os.path.join(base_dir, 'test_res', model_name))

    file_list = os.listdir(os.path.join(base_dir, 'test_data'))
    for file in tqdm(file_list, desc="Processing files"):
        data_path = os.path.join(base_dir, 'test_data', file)
        save_path = os.path.join(base_dir, 'test_res', model_name, file)
        process_file(data_path, save_path, model_name, tokenizer, model, file_config,key_name)

def run_ethics(model_name, model, tokenizer):
    base_dir = 'TrustLLM/Ethics'
    file_config = {
        "emotional_awareness.json": 0.0,
        "explicit_moralchoice_high_ambiguity.json": 1.0,
        "explicit_moralchoice_low_ambiguity.json": 0.0,
        "implicit_ETHICS.json": 0.0,
        "implicit_SocialChemistry101.json": 0.0
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)

def run_privacy(model_name, model, tokenizer):
    base_dir = 'TrustLLM/Privacy'
    file_config = {
        'confAIde_0.json':0.0,
        'privacy_awareness_aug.json': 1.0,
        'privacy_leakage.json': 1.0,
        'privacy_awareness_normal.json': 1.0,
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)

def run_fairness(model_name, model, tokenizer):
    base_dir = 'TrustLLM/Fairness'
    file_config = {
        "disparagement.json": 1.0,
        "preference_force.json": 1.0,
        "preference_plain.json": 1.0,
        "stereotype_agreement.json": 1.0,
        "stereotype_stereoset_classification.json": 0.0,
        'Do_not_answer_stereotype.json':1.0,
    }
    run_task(model_name, model, tokenizer, base_dir, file_config)

def run_misinformation(model_name, model, tokenizer):
    base_dir = 'TrustLLM/Reliability/Misinformation'
    file_config = {
        'external.json': 0.0,
        'hallucination.json': 0.0,
        "golden_advfactuality.json": 1.0,
        "internal.json": 1.0,
        "sycophancy.json": 1.0
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, )
    
def run_ood(model_name, model, tokenizer):
    base_dir = 'TrustLLM/Reliability/OOD'
    file_config = {
        'ood_detection.json':1.0,
        'ood_generalization.json':0.0
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, )

def run_misuse(model_name, model, tokenizer):
    base_dir = 'TrustLLM/Safety/Misuse'
    file_config = {
        # 'do_not_answer.json':1.0,
        'misuse_add.json':1.0,
        'Do-anything-now.json':1.0,
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, )
    
def run_exaggerate(model_name, tokenizer, model):
    base_dir = 'TrustLLM/Safety/exaggerate/'
    file_config = {
        "XSTest.json": 1.0
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, )
    
def run_jailbreak(model_name, tokenizer, model):
    base_dir = 'TrustLLM/Safety/Jailbreak/'
    file_config = {
        'jailbreak_QBB.json':1.0,
        'jailbreak_ITC.json':1.0,
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, )
    
def run_robustness(model_name, tokenizer, model):
    base_dir = 'TrustLLM/NaturalNoise/'
    file_config = {
        'AdvGLUE.json':0.0,
        'AdvInstruction.json':1.0,
        
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, )


def plugin_test_action(model_name, tokenizer, model):
    base_dir = 'plugin/'
    file_config = {
        # 'general_test_few_shot.json':0.0,
        # 'hallucination_prompt_new_few_shot.json':0.0,
        # 'scenario_few_shot.json':0.0,
        # 'general_test_polish.json':0.0,
        # 'general_test_rewritten.json':0.0,
        # 'general_test_llama2.json':0.0,
       #  'refined_plugin_descriptions_list.json':0.0
       'general_test_gpt4.json':0.0,
       'mutli_tool_prompt_force.json': 0.0
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, key_name='action_prompt')
    base_dir = 'plugin/'

def plugin_test(model_name, tokenizer, model):
    base_dir = 'plugin/'
    file_config = {
        'thought_prompt_few_shot.json':0.0,
    }  
    run_task(model_name, model, tokenizer, base_dir, file_config, key_name='thought_prompt')



def jailbreak_test(attack_type, model_name, model, tokenizer):
    result_queue = queue.Queue()
    
    def process_element(el, model_name, tokenizer, index):
        try:
            res = generation(model_name=model_name,model=model, tokenizer=tokenizer, prompt=el['prompt'])
            el['res'] = res
            result_queue.put((index, el))
        except Exception as e:
            print(f"Error processing element at index {index}: {e}")


    def process_file(data_path, save_path, model_name, tokenizer):
        if  'fixed_sentence.json' not in data_path:
            return 0
        # saved_data = []
        
        with open(data_path) as f:
            original_data = json.load(f)
        
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
        else:
            saved_data=original_data
        
        for i in tqdm(range(0, len(saved_data), GROUP_SIZE), desc=f"Processing data for {data_path}", leave=False):

            group_data = saved_data[i:i+GROUP_SIZE]
            threads = []

            for idx, el in enumerate(group_data):
                t = threading.Thread(target=process_element, args=(el, model_name, tokenizer, idx))
                t.start()
                threads.append(t)


            save_json(saved_data, save_path)

            # Wait for all threads to complete
            for t in threads:
                t.join()
                
        save_json(saved_data, save_path)
    file_list = os.listdir('TrustLLM/Safety/Jailbreak/test_data/{}'.format(attack_type))

    for file in tqdm(file_list, desc="Processing files"):
        data_path = os.path.join('TrustLLM/Safety/Jailbreak/test_data/{}'.format(attack_type), file)
        save_path = os.path.join('TrustLLM/Safety/Jailbreak/test_res/{}/{}'.format(model_name, attack_type), file)

        process_file(data_path, save_path, model_name, tokenizer)
    


def compare_json_lengths(data_path, save_path) -> bool:
    # Reading the first JSON file
    with open(data_path, 'r') as file:
        data_json = json.load(file)

    # Check if the save_path file exists
    if not os.path.exists(save_path):
        print(f"'{save_path}' does not exist.")
        return False

    # If the file exists, read it
    with open(save_path, 'r') as file:
        save_json = json.load(file)

    # Comparing the lengths of two JSON files
    are_lengths_equal = len(data_json) == len(save_json)

    return are_lengths_equal



def natural_noise_test(model_name, model, tokenizer, adv_type='advglue'):
    if model_name in online_model:
        GROUP_SIZE=5
    else:
        GROUP_SIZE=1
    result_queue = queue.Queue()
    def process_element_advglue(el,model_name, tokenizer, index):
        res_original = generation(model=model, tokenizer=tokenizer, input=el['original'])
        res_modified = generation(model=model, tokenizer=tokenizer, input=el['modified'])
        return {'original': res_original, 'modified': res_modified, 'label': el['label'], 'task': el['task'], 
                'method': el['method'], 'data construction': el['data construction']}
                    
    def process_element_advinstruction(el,model_name, tokenizer, index):
        try:
            res = generation(model=model, tokenizer=tokenizer, input=el['prompt'])
            el['res'] = res
            result_queue.put((index, el))
        except Exception as e:
            print(f"Error processing element at index {index}: {e}")

            
    def process_file(data_path, save_path, model_name, tokenizer):
 
        saved_data = []
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                saved_data = json.load(f)

        with open(data_path) as f:
            original_data = json.load(f)

        start_idx = len(saved_data)
        remaining_data = original_data[start_idx:]
        
        for i in tqdm(range(0, len(remaining_data), GROUP_SIZE), desc=f"Processing data for {data_path}", leave=False):
            group_data = remaining_data[i:i+GROUP_SIZE]
            threads = []

            for idx, el in enumerate(group_data):
                if adv_type == 'advglue':
                    t = threading.Thread(target=process_element_advglue, args=(el, model_name, tokenizer, idx))

                elif adv_type == 'advinstruction':
                    t = threading.Thread(target=process_element_advinstruction, args=(el, model_name, tokenizer, idx))
                elif adv_type == 'advfact':
                    t = threading.Thread(target=process_element_advinstruction, args=(el, model_name, tokenizer, idx))
                t.start()
                threads.append(t)

            # Collect results for the current group
            group_results = [result_queue.get() for _ in range(len(threads))]
            # Sort results by index to ensure order
            group_results.sort(key=lambda x: x[0])

            # Append sorted results to saved_data
            for _, processed_el in group_results:
                saved_data.append(processed_el)
                
            save_json(saved_data, save_path)

            # Wait for all threads to complete
            for t in threads:
                t.join()

    if adv_type == 'advglue':
        assert os.path.exists('TrustLLM/NaturalNoise/test_res/AdvGLUE/{}'.format(model_name))
        file_list = os.listdir('TrustLLM/NaturalNoise/test_data/AdvGLUE')
        for file in file_list:
            data_path = os.path.join('TrustLLM/NaturalNoise/test_data/AdvGLUE', file)
            save_path = os.path.join('TrustLLM/NaturalNoise/test_res/AdvGLUE/{}'.format(model_name), file)
            if not compare_json_lengths(data_path, save_path):
                process_file(data_path, save_path, model_name, tokenizer, )
                #/home/ubuntu/peft/TrustLLM/NaturalNoise/test_data//golden_advfactuality.json
    elif adv_type == 'advinstruction':
        assert os.path.exists('TrustLLM/NaturalNoise/test_res/AdvInstruction/{}'.format(model_name))
        file_list = os.listdir('TrustLLM/NaturalNoise/test_data/AdvInstruction')
        for file in file_list:
            data_path = os.path.join('TrustLLM/NaturalNoise/test_data/AdvInstruction', file)
            save_path = os.path.join('TrustLLM/NaturalNoise/test_res/AdvInstruction/{}'.format(model_name), file)
            if not compare_json_lengths(data_path, save_path):
                process_file(data_path, save_path, model_name, tokenizer)
    elif adv_type == 'advfact':
        assert os.path.exists('TrustLLM/NaturalNoise/test_res/AdvFactuality/{}'.format(model_name))
        file_list = os.listdir('TrustLLM/NaturalNoise/test_data/AdvFactuality')
        for file in file_list:
            data_path = os.path.join('TrustLLM/NaturalNoise/test_data/AdvFactuality', file)
            save_path = os.path.join('TrustLLM/NaturalNoise/test_res/AdvFactuality/{}'.format(model_name), file)
            if not compare_json_lengths(data_path, save_path):
                process_file(data_path, save_path, model_name, tokenizer)       
        



def run_single_test(args):
    global GROUP_SIZE
    test_type = args.test_type
    model_name=args.model_name 
    print(test_type,args.temperature)
    print(model_name,model_name,model_name,model_name)
    print(online_model)
    if model_name in online_model:
        model = None
        tokenizer = None
    else:
        model, tokenizer = load_model(
            args.model_path,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            revision=args.revision,
            debug=args.debug,
        )

    if test_type == 'natural_noise':
        natural_noise_test(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'robustness':
        run_robustness(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'misinformation':
        run_misinformation(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'fairness':
        run_fairness(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'ethics':      
        run_ethics(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'misuse':
        run_misuse(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'privacy':
        run_privacy(model_name=model_name, model=model, tokenizer=tokenizer,)
    elif test_type == 'exaggerate':
        run_exaggerate(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'jailbreak':
         run_jailbreak(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'ood':
        run_ood(model_name=model_name, model=model, tokenizer=tokenizer)

    elif test_type == 'plugin_test_action':
        plugin_test_action(model_name=model_name, model=model, tokenizer=tokenizer)
    elif test_type == 'plugin_test':
        plugin_test(model_name=model_name, model=model, tokenizer=tokenizer)
    else:
        print("Invalid test_type. Please provide a valid test_type.")
        return None
    return "OK"


@torch.inference_mode()
def main(args,max_retries = 500,retry_interval = 3):
    args.model_name=model_mapping[args.model_path]
        # Generate a unique timestamp for the log file
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_filename = f"logs/log_{args.test_type}_{args.model_name}_test_log_{timestamp}.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for attempt in range(max_retries):
        try:
            state = run_single_test(args)
            message = f"{args.test_type}_{args.model_name}\nTest function successful on attempt {attempt + 1}"
            logging.info(message)
            print(message)
            return state  
        except Exception as e:
            traceback.print_exc()
            message = f"Test function failed on attempt {attempt + 1}:{e}"
            logging.error(message)
            print(message)
            print("Retrying in {} seconds...".format(retry_interval))
            time.sleep(retry_interval)
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
    parser.add_argument("--test_type", type=str, default='plugin')
    args = parser.parse_args()
    state = main(args,)
    print(state)
    
    

