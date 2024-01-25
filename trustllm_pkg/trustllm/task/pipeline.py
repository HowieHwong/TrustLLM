from trustllm.task import ethics, fairness, privacy, robustness, safety, truthfulness
from trustllm.utils import file_process


def run_ethics(explicit_ethics_path=None,
               implicit_ethics_path=None,
               emotional_awareness_path=None):

    evaluator = ethics.EthicsEval()

    explicit_ethics_res_low, explicit_ethics_res_high = None, None
    implicit_ethics_res_ETHICS, implicit_ethics_res_social_norm = None, None
    emotional_awareness_res = None

    if explicit_ethics_path is not None:
        explicit_ethics_data = file_process.load_json(explicit_ethics_path)
        explicit_ethics_res_low = evaluator.explicit_ethics_eval(explicit_ethics_data, eval_type='low')
        explicit_ethics_res_high = evaluator.explicit_ethics_eval(explicit_ethics_data, eval_type='high')

    if implicit_ethics_path is not None:
        implicit_ethics_data = file_process.load_json(implicit_ethics_path)
        implicit_ethics_res_ETHICS = evaluator.implicit_ethics_eval(implicit_ethics_data, eval_type='ETHICS')
        implicit_ethics_res_social_norm = evaluator.implicit_ethics_eval(implicit_ethics_data, eval_type='social_norm')

    if emotional_awareness_path is not None:
        emotional_awareness_data = file_process.load_json(emotional_awareness_path)
        emotional_awareness_res = evaluator.emotional_awareness_eval(emotional_awareness_data)

    return {
        "explicit_ethics_res_low": explicit_ethics_res_low,
        "explicit_ethics_res_high": explicit_ethics_res_high,
        "implicit_ethics_res_ETHICS": implicit_ethics_res_ETHICS,
        "implicit_ethics_res_social_norm": implicit_ethics_res_social_norm,
        "emotional_res": emotional_awareness_res
    }


def run_safety(jailbreak_path=None,
               exaggerated_safety_path=None,
               misuse_path=None,
               toxicity_eval=False,
               toxicity_path=None,
               jailbreak_eval_type="total"):
    evaluator = safety.SafetyEval()

    jailbreak_res, toxicity_res, exaggerated_res, misuse_res = None, None, None, None

    if jailbreak_path is not None:
        jailbreak_data = file_process.load_json(jailbreak_path)
        eval_data, jailbreak_res = evaluator.jailbreak_eval(jailbreak_data, eval_type=jailbreak_eval_type, return_data=True)
        if toxicity_eval and toxicity_path is None:
            toxicity_res = evaluator.toxicity_eval(eval_data)

    if exaggerated_safety_path is not None:
        exaggerated_data = file_process.load_json(exaggerated_safety_path)
        exaggerated_res = evaluator.exaggerated_eval(exaggerated_data)

    if misuse_path is not None:
        misuse_data = file_process.load_json(misuse_path)
        misuse_res = evaluator.misuse_eval(misuse_data)

    if toxicity_eval and toxicity_path is not None:
        toxicity_data = file_process.load_json(toxicity_path)  # load eval data for toxicity evaluation
        toxicity_res = evaluator.toxicity_eval(toxicity_data)

    return {
        "jailbreak_res": jailbreak_res,
        "exaggerated_safety_res": exaggerated_res,
        "misuse_res": misuse_res,
        "toxicity_res": toxicity_res
    }



def run_robustness():
    pass


def run_privacy():
    pass


def run_truthfulness():
    pass


def run_fairness():
    pass