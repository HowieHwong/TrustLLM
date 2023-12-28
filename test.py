from trustllm.task import safety
from trustllm.utils import file_process
from trustllm import config

evaluator = safety.SafetyEval()

jailbreak_data = file_process.load_json('jailbreak_data.json')
print(evaluator.jailbreak_eval(jailbreak_data, eval_type='total'))
print(evaluator.jailbreak_eval(jailbreak_data, eval_type='single'))

exaggerated_data = file_process.load_json('exaggerated_data.json')
print(evaluator.exaggerated_eval(exaggerated_data))

toxicity_data = file_process.load_json('toxicity_data.json')
print(evaluator.toxicity_eval(toxicity_data, perspective_api=config.perspective_key))

misuse_data = file_process.load_json('misuse_data.json')
print(evaluator.misuse_eval(misuse_data))



