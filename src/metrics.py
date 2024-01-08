# metrics.py
from sklearn.metrics import confusion_matrix

def calculate_accuracy(approved_count, total_predictions):
    if total_predictions == 0:
        return 0.0  # or any default value you prefer
    else:
        accuracy = approved_count / total_predictions
        return accuracy

def calculate_metrics(true_positives, true_negatives, false_positives, false_negatives):
    # Accuracy
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    # False Acceptance Rate (FAR)
    denominator_far = false_positives + true_negatives
    far = false_positives / denominator_far if denominator_far != 0 else 0

    # False Rejection Rate (FRR)
    denominator_frr = false_negatives + true_positives
    frr = false_negatives / denominator_frr if denominator_frr != 0 else 0

    # Precision
    denominator_precision = true_positives + false_positives
    precision = true_positives / denominator_precision if denominator_precision != 0 else 0

    # Recall
    denominator_recall = true_positives + false_negatives
    recall = true_positives / denominator_recall if denominator_recall != 0 else 0

    # F1 Score
    denominator_f1 = precision + recall
    f1_score = 2 * (precision * recall) / denominator_f1 if denominator_f1 != 0 else 0

    return accuracy, far, frr, precision, recall, f1_score
