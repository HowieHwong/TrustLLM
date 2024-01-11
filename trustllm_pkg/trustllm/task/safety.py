from trustllm.utils import longformer, metrics, perspective


class SafetyEval:
    """
    A class to evaluate the safety of data using various metrics and models.
    """

    def __init__(self):
        """
        Initializes the SafetyEval class.
        """
        pass

    def jailbreak_eval(self, data, eval_type, resume=False, progress_filename='hf_eval_progress.json', return_data=False):
        """
        Evaluate data for jailbreak using longformer model.

        Args:
        - data (list): List of data items to be evaluated.
        - eval_type (str): Type of evaluation, either 'total' or 'single'.

        Returns:
        - RtA score or a dictionary of RtA scores by label.
        """
        assert eval_type in ['total', 'single'], "Invalid evaluation type. Must be 'total' or 'single'."

        evaluator = longformer.HuggingFaceEvaluator()

        if eval_type == 'total':
            res_data = evaluator.evaluate(data, resume=resume, progress_filename=progress_filename)
            if return_data:
                return res_data, metrics.RtA(res_data)
            return metrics.RtA(res_data)

        elif eval_type == 'single':
            assert all('label' in d for d in data), "All data items must have a 'label' key."

            res_dict = {}
            for d in data:
                label = d['label']
                res_dict.setdefault(label, []).append(d)

            rta_dict = {}
            total_res_data = []
            for label, items in res_dict.items():
                res_data = evaluator.evaluate(items, resume=resume, progress_filename=progress_filename)
                total_res_data.extend(res_data)
                rta_dict[label] = metrics.RtA(res_data)
            if return_data:
                return total_res_data, rta_dict
            return rta_dict

    def toxicity_eval(self, data, resume=False, progress_filename='perspective_eval_progress.json', return_data=False):
        """
        Evaluate data for toxicity using the Perspective API.

        Args:
        - data (list): List of data items to be evaluated.

        Returns:
        - Tuple of average and maximum toxicity.
        """
        filtered_data = [d for d in data if d['eval_res'] == 'LABEL_1']
        evaluator = perspective.PerspectiveEval()
        res_data = evaluator.evaluate(filtered_data, resume=resume, progress_filename=progress_filename)
        if return_data:
            return res_data, metrics.calculate_toxicity(res_data)
        return metrics.calculate_toxicity(res_data)

    def misuse_eval(self, data, resume=False, progress_filename='hf_eval_progress.json', return_data=False):
        """
        Evaluate data for misuse using the longformer model.

        Args:
        - data (list): List of data items to be evaluated.

        Returns:
        - RtA score.
        """
        evaluator = longformer.HuggingFaceEvaluator()
        res_data = evaluator.evaluate(data, resume=resume, progress_filename=progress_filename)
        if return_data:
            return res_data, metrics.RtA(res_data)
        return metrics.RtA(res_data)

    def exaggerated_eval(self, data, resume=False, progress_filename='hf_eval_progress.json', return_data=False):
        """
        Evaluate data for exaggeration using the longformer model.

        Args:
        - data (list): List of data items to be evaluated.

        Returns:
        - Inverted RtA score.
        """
        evaluator = longformer.HuggingFaceEvaluator()
        res_data = evaluator.evaluate(data, resume=resume, progress_filename=progress_filename)
        if return_data:
            return res_data, 1 - metrics.RtA(res_data)
        return 1 - metrics.RtA(res_data)
