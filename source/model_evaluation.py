# model_evaluation.py
def compare_models(model_results):
    """
    Compare multiple models' results.
    
    Args:
        model_results (dict): Dictionary containing model names as keys and their AUC scores as values.
    """
    print("\nComparison of Model Results:")
    for model_name, auc_score in model_results.items():
        print(f"{model_name} ROC AUC: {auc_score}")

