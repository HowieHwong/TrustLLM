
from trustllm.task import safety,fairness,truthfulness
from trustllm.utils import file_process
from trustllm import config




from trustllm.generation.generation import LLMGeneration
from trustllm.utils import file_process
from trustllm import config

import concurrent.futures

config.replicate_api= "r8_EBJDSqZ6KDR2EmN0NWRodPva3018ceS0nd6BF"

def run_task(test_type):
    run_robu=LLMGeneration(test_type=test_type,
                        online_model=True,
                        use_replicate=True,
                model_path='meta/meta-llama-3-8b-instruct',
                data_path='/Users/admin/Documents/dataset')
    run_robu.generation_results()

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(run_task,['safety','fairness','truthfulness','privacy','robustness','ethics'])