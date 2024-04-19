
from trustllm.task import safety,fairness,truthfulness
from trustllm.utils import file_process
from trustllm import config




from trustllm.generation.generation import LLMGeneration
from trustllm.utils import file_process
from trustllm import config

config.replicate_api= "r8_EBJDSqZ6KDR2EmN0NWRodPva3018ceS0nd6BF"
run_robu=LLMGeneration(test_type='fairness',
                       online_model=True,
                       use_replicate=True,
              model_path='meta/meta-llama-3-70b-instruct',
              data_path='/Users/admin/Documents/dataset')
run_robu.generation_results()