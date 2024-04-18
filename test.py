
from trustllm.task import safety,fairness,truthfulness
from trustllm.utils import file_process
from trustllm import config




config.openai_key= "1f462c580d06407eb49954553ab22ff7"

evaluator = truthfulness.TruthfulnessEval()
internal= file_process.load_json('/Users/admin/Downloads/mixtral-8x7B/truthfulness/internal.json')

evaluator.eval_internal_squad(internal)