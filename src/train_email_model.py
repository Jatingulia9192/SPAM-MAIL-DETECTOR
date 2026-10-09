
import email
import re
from pathlib import Path
from email import policy

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "email_raw"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "email_spam_model.joblib"


def extract_email_text(file_path):
    try:
        with open(file_path, "rb") as file:
            message = email.message_from_binary_file(
                file, policy=policy.default
            )

        subject = str(message.get("subject", ""))
        body_parts = []

        if message.is_multipart():
            for part in message.walk():
                if part.get_content_maintype() != "text":
                    continue
                if part.get_content_subtype() not in ("plain", "html"):
                    continue
                try:
                    content = part.get_content()
                except Exception:
                    continue
                if isinstance(content, str):
                    body_parts.append(content)
        else:
            try:
                content = message.get_content()
                if isinstance(content, str):
                    body_parts.append(content)
            except Exception:
                pass

        body = " ".join(body_parts)
        text = f"{subject} {body}"
        text = re.sub(r"\s+", " ", text).strip()
        return text

    except Exception as error:
        print(f"Skipped {file_path.name}: {error}")
        return ""


def load_dataset():
    records = []

    folders = [
        ("easy_ham", 0),
        ("spam", 1),
    ]

    for folder_name, label in folders:
        folder = DATA_DIR / folder_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue

            text = extract_email_text(file_path)

            if text:
                records.append({
                    "text": text,
                    "label": label,
                })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.len() > 0]

    return df


def main():
    print("Loading email dataset...")
    df = load_dataset()

    print(f"\nUsable emails: {len(df)}")
    print("\nClass counts:")
    print(df["label"].map({0: "ham", 1: "spam"}).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
            ),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining email model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": model,
        "model_name": "Logistic Regression with TF-IDF",
        "dataset": "Apache SpamAssassin Public Corpus",
        "test_auc": float(roc_auc_score(y_test, probabilities)),
        "test_size": len(X_test),
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved email model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
