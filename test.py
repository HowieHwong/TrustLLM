from trustllm.task import fairness
from trustllm.utils import file_process
from trustllm import config

evaluator = fairness.FairnessEval()

stereotype_recognition_data = file_process.load_json('stereotype_recognition_data_json_path')
evaluator.stereotype_recognition_eval(stereotype_recognition_data)

stereotype_agreement_data = file_process.load_json('stereotype_agreement_data_json_path')
evaluator.stereotype_agreement_eval(stereotype_agreement_data)

stereotype_query_test_data = file_process.load_json('stereotype_query_test_data_json_path')
evaluator.stereotype_query_eval(stereotype_query_test_data)


disparagement_data = file_process.load_json('disparagement_data_json_path')
evaluator.disparagement_eval(disparagement_data)

preference_data = file_process.load_json('preference_data_json_path')
evaluator.preference_eval(preference_data)