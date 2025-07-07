import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'status' not in df.columns:
        raise ValueError("Missing 'status' column in dataset")

    X = df.drop(columns=['status', 'name', 'filename'], errors='ignore')
    y = df['status']
    return X, y, list(X.columns), df


def run_rfe_models(X, y, feature_names, n_features=5):
    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        selected = [feature_names[i] for i, flag in enumerate(rfe.support_) if flag]
        ranking = [(feature_names[i], r) for i, r in enumerate(rfe.ranking_)]

        model.fit(rfe.transform(X_train), y_train)
        y_pred = model.predict(rfe.transform(X_test))
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            'rfe': rfe,
            'selected_features': selected,
            'ranking': sorted(ranking, key=lambda x: x[1]),
            'accuracy': acc,
            'estimator': model
        }

    return results


def plot_rankings(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        features = [f for f, _ in data['ranking']]
        ranks = [r for _, r in data['ranking']]
        ax.barh(features, ranks, color=['green' if r == 1 else 'red' for r in ranks])
        ax.set_title(f"{model_name} Rankings")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(results):
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color='skyblue')
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.3f}", ha='center')
    plt.tight_layout()
    plt.show()


def save_results_txt(results, consensus_features, output_file='selected_features.txt'):
    with open(output_file, 'w') as f:
        f.write("FEATURE SELECTION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        sorted_acc = sorted([(name, res['accuracy']) for name, res in results.items()], key=lambda x: -x[1])
        f.write("Model Accuracy Ranking:\n")
        for i, (name, acc) in enumerate(sorted_acc, 1):
            f.write(f"{i}. {name}: {acc:.4f}\n")

        f.write("\nConsensus Features (selected by multiple models):\n")
        for feat in consensus_features:
            f.write(f"- {feat}\n")

        f.write("\nAll Model Selected Features:\n")
        for name, res in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"  Selected Features: {', '.join(res['selected_features'])}\n")

    print(f"Results saved to {output_file}")


def extract_consensus(results):
    counter = {}
    for res in results.values():
        for feat in res['selected_features']:
            counter[feat] = counter.get(feat, 0) + 1
    return [feat for feat, count in counter.items() if count > 1]


def main():
    X, y, feature_names, _ = load_data('dataset/augmented_dataset.csv')
    results = run_rfe_models(X, y, feature_names)

    for name, res in results.items():
        print(f"{name}: {res['accuracy']:.4f}, Features: {res['selected_features']}")

    consensus = extract_consensus(results)
    print(f"\nConsensus features (selected by multiple models): {consensus}")

    plot_rankings(results)
    plot_accuracy_comparison(results)
    save_results_txt(results, consensus)


if __name__ == '__main__':
    main()
