"""Original benchmark filenames and sampling temperatures."""

TASKS = {
    "ethics": {
        "awareness.json": 0.0,
        "explicit_moralchoice.json": 1.0,
        "implicit_ETHICS.json": 0.0,
        "implicit_SocialChemistry101.json": 0.0,
    },
    "privacy": {
        "privacy_awareness_confAIde.json": 0.0,
        "privacy_awareness_query.json": 1.0,
        "privacy_leakage.json": 1.0,
    },
    "fairness": {
        "disparagement.json": 1.0,
        "preference.json": 1.0,
        "stereotype_agreement.json": 1.0,
        "stereotype_query_test.json": 1.0,
        "stereotype_recognition.json": 0.0,
    },
    "truthfulness": {
        "external.json": 0.0,
        "hallucination.json": 0.0,
        "golden_advfactuality.json": 1.0,
        "internal.json": 1.0,
        "sycophancy.json": 1.0,
    },
    "robustness": {
        "ood_detection.json": 1.0,
        "ood_generalization.json": 0.0,
        "AdvGLUE.json": 0.0,
        "AdvInstruction.json": 1.0,
    },
    "safety": {"jailbreak.json": 1.0, "exaggerated_safety.json": 1.0, "misuse.json": 1.0},
}
