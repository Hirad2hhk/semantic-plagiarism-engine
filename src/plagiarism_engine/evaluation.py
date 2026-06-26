def calculate_precision(true_positives: int, 
                        predicted_positives: int) -> float:
    return true_positives / (predicted_positives + 1e-8)

def calculate_recall(true_positives: int, 
                     actual_positives: int) -> float:
    return true_positives / (actual_positives + 1e-8)

def f1_score(precision: float, recall: float) -> float:
    return 2 * (precision*recall)/(precision + recall + 1e-8)
