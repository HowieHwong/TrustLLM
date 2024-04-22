import os
import difflib
import inspect
from trustllm.task import ethics, fairness, privacy, robustness, safety, truthfulness
from trustllm.utils import file_process
import traceback

def run_ethics(
    all_folder_path=None,
    explicit_ethics_path=None, 
    implicit_ethics_path_social_norm=None, 
    implicit_ethics_path_ETHICS=None, 
    awareness_path=None
):
    if all_folder_path:
        explicit_ethics_path = os.path.join(all_folder_path, "explicit_moralchoice.json")
        implicit_ethics_path_social_norm = os.path.join(all_folder_path, "implicit_SocialChemistry101.json")
        implicit_ethics_path_ETHICS = os.path.join(all_folder_path, "implicit_ETHICS.json")
        #awareness_path = os.path.join(all_folder_path, "awareness.json")
    
    evaluator = ethics.EthicsEval()
    explicit_ethics_res_low, explicit_ethics_res_high = None, None
    implicit_ethics_res_ETHICS, implicit_ethics_res_social_norm = None, None
    awareness_res = None
# The above Python code snippet is checking if certain paths (`explicit_ethics_path` and
# `implicit_ethics_path_social_norm`) are not None, and if so, it loads JSON data from those paths
# using `file_process.load_json` function.

    if explicit_ethics_path is not None:
        explicit_ethics_data = file_process.load_json(explicit_ethics_path)
        explicit_ethics_res_low = evaluator.explicit_ethics_eval(
            explicit_ethics_data, eval_type="low"
        )
        
        explicit_ethics_res_high = evaluator.explicit_ethics_eval(
            explicit_ethics_data, eval_type="high"
        )
        print('explicit_ethics_res_low',explicit_ethics_res_low,'explicit_ethics_res_high' ,explicit_ethics_res_high)

    if implicit_ethics_path_social_norm is not None:
        implicit_ethics_data_social_norm = file_process.load_json(implicit_ethics_path_social_norm)
        implicit_ethics_res_ETHICS = evaluator.implicit_ethics_eval(
            implicit_ethics_data_social_norm, eval_type="social_norm"
        )
        print('implicit_ethics_res_ETHICS',implicit_ethics_res_ETHICS)

    if implicit_ethics_path_ETHICS is not None:
        implicit_ethics_data_ETHICS = file_process.load_json(implicit_ethics_path_ETHICS)
        implicit_ethics_res_social_norm = evaluator.implicit_ethics_eval(
            implicit_ethics_data_ETHICS, eval_type="ETHICS"
        )
        print("implicit_ethics_res_social_norm",implicit_ethics_res_social_norm)

    if awareness_path is not None:
        awareness_data = file_process.load_json(awareness_path)
        awareness_res = evaluator.awareness_eval(
            awareness_data
        )

    return {
        "explicit_ethics_res_low": explicit_ethics_res_low,
        "explicit_ethics_res_high": explicit_ethics_res_high,
        "implicit_ethics_res_ETHICS": implicit_ethics_res_ETHICS,
        "implicit_ethics_res_social_norm": implicit_ethics_res_social_norm,
        "emotional_res": awareness_res,
    }


def run_safety(
    all_folder_path=None,
    jailbreak_path=None,
    exaggerated_safety_path=None,
    misuse_path=None,
    toxicity_eval=False,
    toxicity_path=None,
    jailbreak_eval_type="total",
):
    
    if all_folder_path:
        jailbreak_path = os.path.join(all_folder_path, "jailbreak.json")
        exaggerated_safety_path = os.path.join(all_folder_path, "exaggerated_safety.json")
        misuse_path = os.path.join(all_folder_path, "misuse.json")
        #if toxicity_eval:
        #toxicity_path = os.path.join(all_folder_path, "jailbreak.json")
    
    evaluator = safety.SafetyEval()

    jailbreak_res, toxicity_res, exaggerated_res, misuse_res = None, None, None, None

    if jailbreak_path is not None:
        jailbreak_data = file_process.load_json(jailbreak_path)
        eval_data, jailbreak_res = evaluator.jailbreak_eval(
            jailbreak_data, eval_type=jailbreak_eval_type, return_data=True,resume=True
        )
        if toxicity_eval and toxicity_path is None:
            toxicity_res = evaluator.toxicity_eval(eval_data,resume=True)

    if exaggerated_safety_path is not None:
        exaggerated_data = file_process.load_json(exaggerated_safety_path)
        exaggerated_res = evaluator.exaggerated_eval(exaggerated_data)
    print(misuse_path)
    if misuse_path is not None:
        misuse_data = file_process.load_json(misuse_path)
        misuse_res = evaluator.misuse_eval(misuse_data)

    if toxicity_eval and toxicity_path is not None:
        toxicity_data = file_process.load_json(
            toxicity_path
        )  # load eval data for toxicity evaluation
        toxicity_res = evaluator.toxicity_eval(toxicity_data)

    return {
        "jailbreak_res": jailbreak_res,
        "exaggerated_safety_res": exaggerated_res,
        "misuse_res": misuse_res,
        "toxicity_res": toxicity_res,
    }


def run_robustness(
    all_folder_path=None,
    advglue_path=None,
    advinstruction_path=None,
    ood_detection_path=None,
    ood_generalization_path=None,
):
    if all_folder_path:
        advglue_path = os.path.join(all_folder_path, "AdvGLUE.json")
        advinstruction_path = os.path.join(all_folder_path, "AdvInstruction.json")
        ood_detection_path = os.path.join(all_folder_path, "ood_detection.json")
        ood_generalization_path = os.path.join(all_folder_path, "ood_generalization.json")
    
    evaluator = robustness.RobustnessEval()

    advglue_res, advinstruction_res, ood_detection_res, ood_generalization_res = (
        None,
        None,
        None,
        None,
    )

    if advglue_path is not None:
        advglue_data = file_process.load_json(advglue_path)
        advglue_res = evaluator.advglue_eval(advglue_data)

    if advinstruction_path is not None:
        advinstruction_data = file_process.load_json(advinstruction_path)
        advinstruction_res = evaluator.advinstruction_eval(advinstruction_data)

    if ood_detection_path is not None:
        ood_detection_data = file_process.load_json(ood_detection_path)
        ood_detection_res = evaluator.ood_detection(ood_detection_data)

    if ood_generalization_path is not None:
        ood_generalization_data = file_process.load_json(ood_generalization_path)
        ood_generalization_res = evaluator.ood_generalization(ood_generalization_data)

    return {
        "advglue_res": advglue_res,
        "advinstruction_res": advinstruction_res,
        "ood_detection_res": ood_detection_res,
        "ood_generalization_res": ood_generalization_res,
    }


def run_privacy(
    all_folder_path=None,
    privacy_confAIde_path=None,
    privacy_awareness_query_path=None,
    privacy_leakage_path=None,
):
        
    if all_folder_path:
        privacy_confAIde_path = os.path.join(all_folder_path, "privacy_awareness_confAIde.json")
        privacy_awareness_query_path = os.path.join(all_folder_path, "privacy_awareness_query.json")
        privacy_leakage_path = os.path.join(all_folder_path, "privacy_leakage.json")

    evaluator = privacy.PrivacyEval()
    

    (
        privacy_confAIde_res,
        privacy_awareness_query_normal_res,
        privacy_awareness_query_aug_res,
        privacy_leakage_res,
    ) = (
        None,
        None,
        None,
        None,
    )

    if privacy_confAIde_path is not None:
        privacy_confAIde_data = file_process.load_json(privacy_confAIde_path)
        privacy_confAIde_res = evaluator.ConfAIDe_eval(privacy_confAIde_data)

    if privacy_awareness_query_path is not None:
        privacy_awareness_query_data = file_process.load_json(
            privacy_awareness_query_path
        )
        privacy_awareness_query_normal_res = evaluator.awareness_query_eval(
            privacy_awareness_query_data, type="normal"
        )
        privacy_awareness_query_aug_res = evaluator.awareness_query_eval(
            privacy_awareness_query_data, type="aug"
        )

    if privacy_leakage_path is not None:
        privacy_leakage_data = file_process.load_json(privacy_leakage_path)
        privacy_leakage_res = evaluator.leakage_eval(privacy_leakage_data)

    return {
        "privacy_confAIde": privacy_confAIde_res,
        "privacy_awareness_query_normal": privacy_awareness_query_normal_res,
        "privacy_awareness_query_aug": privacy_awareness_query_aug_res,
        "privacy_leakage": privacy_leakage_res,
    }


def run_truthfulness(
    all_folder_path=None,
    internal_path=None,
    external_path=None,
    hallucination_path=None,
    sycophancy_path=None,
    advfact_path=None,
):
    if all_folder_path:
        internal_path = os.path.join(all_folder_path, "internal.json")
        external_path = os.path.join(all_folder_path, "external.json")
        hallucination_path = os.path.join(all_folder_path, "hallucination.json")
        sycophancy_path = os.path.join(all_folder_path, "sycophancy.json")
        advfact_path = os.path.join(all_folder_path, "golden_advfactuality.json")

    evaluator = truthfulness.TruthfulnessEval()

    (
        internal_res,
        external_res,
        hallucination_res,
        sycophancy_persona_res,
        sycophancy_preference_res,
        advfact_res,
    ) = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    try:
        if internal_path is not None:
            internal_data = file_process.load_json(internal_path)
            internal_res = evaluator.internal_eval(internal_data)

        if external_path is not None:
            external_data = file_process.load_json(external_path)
            external_res = evaluator.external_eval(external_data)

        if hallucination_path is not None:
            hallucination_data = file_process.load_json(hallucination_path)
            hallucination_res = evaluator.hallucination_eval(hallucination_data)

        if sycophancy_path is not None:
            sycophancy_data = file_process.load_json(sycophancy_path)
            sycophancy_persona_res = evaluator.sycophancy_eval(
                sycophancy_data, eval_type="persona"
            )
            sycophancy_preference_res = evaluator.sycophancy_eval(
                sycophancy_data, eval_type="preference"
            )

        if advfact_path is not None:
            advfact_data = file_process.load_json(advfact_path)
            advfact_res = evaluator.advfact_eval(advfact_data)
    except Exception as e:
        traceback.print_exc();
        print(e)
    return {
        "misinformation_internal": internal_res,
        "misinformation_external": external_res,
        "hallucination": hallucination_res,
        "sycophancy_persona": sycophancy_persona_res,
        "sycophancy_preference": sycophancy_preference_res,
        "advfact": advfact_res,
    }



def find_best_match(keyword, file_list):
    """Find the best match for a keyword in a list of filenames."""
    matches = difflib.get_close_matches(keyword, file_list, n=1, cutoff=0.1)
    return matches[0] if matches else None

def auto_assign_paths(all_folder_path, param_names):
    """Automatically assign paths based on parameter names and files in the given folder."""
    files = os.listdir(all_folder_path)
    paths = {}
    for name in param_names:
        # Convert parameter name to expected file name pattern
        key = name.replace('_path', '')
        expected_filename = f"{key}.json"
        matched_file = find_best_match(expected_filename, files)
        if matched_file:
            paths[name] = os.path.join(all_folder_path, matched_file)
    return paths

def run_fairness(
    all_folder_path=None,
    stereotype_recognition_path=None,
    stereotype_agreement_path=None,
    stereotype_query_test_path=None,
    disparagement_path=None,
    preference_path=None,
):

    if all_folder_path:
        stereotype_recognition_path = os.path.join(all_folder_path, "stereotype_recognition.json")
        stereotype_agreement_path = os.path.join(all_folder_path, "stereotype_agreement.json")
        stereotype_query_test_path = os.path.join(all_folder_path, "stereotype_query_test.json")
        disparagement_path = os.path.join(all_folder_path, "disparagement.json")
        preference_path = os.path.join(all_folder_path, "preference.json")

    evaluator = fairness.FairnessEval()

    (
        stereotype_recognition_res,
        stereotype_agreement_res,
        stereotype_query_res,
        disparagement_res,
        preference_res,
    ) = (None, None, None, None, None)

    if stereotype_recognition_path is not None:
        stereotype_recognition_data = file_process.load_json(
            stereotype_recognition_path
        )
        stereotype_recognition_res = evaluator.stereotype_recognition_eval(
            stereotype_recognition_data
        )

    if stereotype_agreement_path is not None:
        stereotype_agreement_data = file_process.load_json(stereotype_agreement_path)
        stereotype_agreement_res = evaluator.stereotype_agreement_eval(
            stereotype_agreement_data
        )

    if stereotype_query_test_path is not None:
        stereotype_query_data = file_process.load_json(stereotype_query_test_path)
        stereotype_query_res = evaluator.stereotype_query_eval(stereotype_query_data)

    if disparagement_path is not None:
        disparagement_data = file_process.load_json(disparagement_path)
        disparagement_res = evaluator.disparagement_eval(disparagement_data)

    if preference_path is not None:
        preference_data = file_process.load_json(preference_path)
        preference_res = evaluator.preference_eval(preference_data)

    return {
        "stereotype_recognition": stereotype_recognition_res,
        "stereotype_agreement": stereotype_agreement_res,
        "stereotype_query": stereotype_query_res,
        "disparagement": disparagement_res,
        "preference": preference_res,
    }
