import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, f1_score


def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def make_pipe(model):
    # text ko numbers mein badalna aur model, ek hi pipeline mein
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)),
        ("model", model),
    ])


def compare_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in get_models().items():
        s = cross_validate(make_pipe(model), X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "Model": name,
            "Accuracy": s["test_accuracy"].mean(),
            "Precision": s["test_precision"].mean(),
            "Recall": s["test_recall"].mean(),
            "F1": s["test_f1"].mean(),
            "ROC-AUC": s["test_roc_auc"].mean(),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def word_weights(pipe):
    # har word spam ki taraf kitna dhakelta hai (positive = spam)
    model = pipe.named_steps["model"]
    words = pipe.named_steps["tfidf"].get_feature_names_out()
    if hasattr(model, "coef_"):
        w = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        w = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return None
    return {word: float(x) for word, x in zip(words, w)}


def evaluate_and_save(results_df, X_train, X_test, y_train, y_test, tag="spam",
                      labels=("Not Spam", "Spam")):
    best_name = results_df.iloc[0]["Model"]
    pipe = make_pipe(get_models()[best_name])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Best model:", best_name)
    print(classification_report(y_test, y_pred, target_names=list(labels)))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=list(labels))
    plt.title(f"{tag} - Confusion Matrix")
    plt.savefig(f"../assets/{tag}_confusion.png", bbox_inches="tight")
    plt.show()

    weights = word_weights(pipe)
    if weights:
        s = pd.Series(weights).sort_values()
        top = pd.concat([s.head(10), s.tail(10)])
        top.plot(kind="barh", figsize=(8, 6))
        plt.title("Words pushing towards Not Spam (left) and Spam (right)")
        plt.savefig(f"../assets/{tag}_top_words.png", bbox_inches="tight")
        plt.show()

    results_df.round(3).to_csv(f"../assets/{tag}_results.csv", index=False)
    proba = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    test_f1 = f1_score(y_test, y_pred)

    joblib.dump({
        "pipeline": pipe,
        "model_name": best_name,
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "weights": weights,
    }, f"../models/{tag}_model.joblib")
    print(f"Saved models/{tag}_model.joblib | Test ROC-AUC: {test_auc:.3f} | Test F1: {test_f1:.3f}")
    return pipe