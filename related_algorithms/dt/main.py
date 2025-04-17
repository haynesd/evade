import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from data_loader import getDataFromCSV
from scipy.stats import mode


def run_supervised(X_train, y_train, X_test, y_test):
    print("\n--- Supervised Learning: Decision Tree (J48 equivalent) ---")
    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(
        y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))


def run_unsupervised(X_train, X_test, y_test):
    print("\n--- Unsupervised Learning: Expectation-Maximization (EM) ---")
    em = GaussianMixture(n_components=2, random_state=42)
    em.fit(X_train)
    y_pred = em.predict(X_test)

    label_map = {}
    for cluster in set(y_pred):
        indices = [i for i, label in enumerate(y_pred) if label == cluster]
        if indices:
            true_labels = y_test.iloc[indices]
            mode_val = mode(true_labels, keepdims=True).mode
            label_map[cluster] = int(mode_val[0]) if len(mode_val) > 0 else 0
        else:
            label_map[cluster] = 0

    y_pred_mapped = [label_map[label] for label in y_pred]

    print("Classification Report:\n", classification_report(
        y_test, y_pred_mapped, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mapped))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_mapped))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to the CSV data file')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = getDataFromCSV(args.csv)
    run_supervised(X_train, y_train, X_test, y_test)
    run_unsupervised(X_train, X_test, y_test)


if __name__ == "__main__":
    main()
