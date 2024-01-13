import os, json
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import google.generativeai as palm
from google.generativeai.types import safety_types
import openai
import traceback
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
from trustllm.utils import file_process
import trustllm

online_model = file_process.load_json('trustllm/prompt/model_info.json')['online_model']
deepinfra_model = file_process.load_json('trustllm/prompt/model_info.json')['deep_model']
model_mapping = file_process.load_json('trustllm/prompt/model_info.json')['model_mapping']
rev_model_mapping = {value: key for key, value in model_mapping.items()}


def get_models():
    return model_mapping, online_model


def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=H4PeMykZWAvZsatz5vxSpLNG&client_secret=twdirokNnADKzeUIpPQ9Zl48OeSz4XpO"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def get_ernie_res(string, temperature):
    if (temperature == 0.0):
        temperature = 0.00000001
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": string,
            }
        ],
        'temperature': temperature
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    res_data = json.loads(response.text)
    result_text = res_data.get('result', '')
    return result_text


def get_res_chatgpt(string, gpt_model, temperature):
    if gpt_model == 'gpt-3.5-turbo':
        openai.api_key = trustllm.config.openai_key
    elif gpt_model == "TODO":
        openai.api_key = trustllm.config.openai_key
    else:
        raise ValueError('No support model!')
    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "user",
             "content": string}
        ],
        temperature=temperature,
    )

    if not completion.choices[0].message['content']:
        raise ValueError("The response from the API is NULL or an empty string!")

    return completion.choices[0].message['content']


class DeepInfraAPI:
    def __init__(self, auth_token):
        self.base_url = "https://api.deepinfra.com/v1/inference/{model}"
        self.headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    def create(self, model, text, max_tokens, temperature):
        payload = {
            "input": text,
            "max_new_tokens": max_tokens,
            "temperature": temperature,

        }

        try:
            response = requests.post(self.base_url.format(model=model), json=payload, headers=self.headers, timeout=1)
            response.raise_for_status()  # Raises HTTPError for bad responses
            response_json = response.json()
            results = response_json.get('results', [])
            return results[0].get('generated_text') if results else None
        except Exception:
            traceback.print_exc()
            return None


def deepinfra_api(string, model, temperature):
    api_token = trustllm.config.deepinfra_api
    rev_mapping = {'dolly-12b': 'databricks/dolly-v2-12b', }
    if model in ['dolly-12b']:
        api = DeepInfraAPI(api_token)
        response_content = api.create(
            model=rev_mapping[model],
            text=string,
            max_tokens=4000,
            temperature=temperature,
        )
        return response_content
    else:
        openai.api_key = api_token
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        top_p = 1 if temperature <= 1e-5 else 0.9
        print(temperature)
        chat_completion = openai.ChatCompletion.create(
            model=rev_model_mapping[model],
            messages=[{"role": "user", "content": string}],
            max_tokens=5192,
            temperature=temperature,
            top_p=top_p,
        )
        return chat_completion.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def claude_api(string, model, temperature):
    anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=trustllm.config.claude_api,
    )

    completion = anthropic.completions.create(
        model=model,  # "claude-2", "claude-instant-1"
        max_tokens_to_sample=4000,
        temperature=temperature,
        prompt=f"{HUMAN_PROMPT} {string}{AI_PROMPT}", )

    # print(chat_completion.choices[0].message.content)
    return completion.completion


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def palm_api(string, model, temperature):
    palm.configure(api_key=trustllm.config.palm_api)

    model_mapping = {
        'bison-001': 'models/text-bison-001',
        }
    completion = palm.generate_text(
        model=model_mapping[model],  # models/text-bison-001
        prompt=string,
        temperature=temperature,
        # The maximum length of the response
        max_output_tokens=4000,


        safety_settings=[

            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },

            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },

            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },

            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]
    )
    return completion.result


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
def gen_online(model_name, prompt, temperature):
    if model_name == 'ernie':
        res = get_ernie_res(prompt, temperature=temperature)
    elif model_name == 'chatgpt':
        res = get_res_chatgpt(prompt, gpt_model='gpt-3.5-turbo', temperature=temperature)
    elif model_name == 'gpt-4':
        res = get_res_chatgpt(prompt, gpt_model='gpt-4', temperature=temperature)
    elif model_name in deepinfra_model:
        print(model_name)
        res = deepinfra_api(prompt, model=model_name, temperature=temperature)
    elif model_name in ['claude-2']:
        res = claude_api(prompt, model=model_name, temperature=temperature)
    elif model_name == 'bison-001':
        res = palm_api(prompt, model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return res


