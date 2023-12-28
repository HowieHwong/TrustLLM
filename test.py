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
evaluator.preference_eval(preference_data, type='plain')
evaluator.preference_eval(preference_data, type='force')


from trustllm.task import robustness
from trustllm.utils import file_process
from trustllm import config

evaluator = robustness.RobustnessEval()

advglue_data = file_process.load_json('advglue_data_json_path')
print(evaluator.advglue_eval(advglue_data))

advinstruction_data = file_process.load_json('advinstruction_data_json_path')
print(evaluator.advglue_eval(advinstruction_data))

ood_detection_data = file_process.load_json('ood_detection_data_json_path')
print(evaluator.ood_detection(ood_detection_data))

ood_generalization_data = file_process.load_json('ood_generalization_data_json_path')
print(evaluator.ood_generalization(ood_generalization_data))

