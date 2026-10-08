import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Classifier")
st.caption(
    "Trained on the UCI SMS Spam Collection. Works on short text messages, "
    "so results on long emails may be less reliable."
)


@st.cache_resource
def load_bundle():
    return joblib.load("models/spam_model.joblib")


bundle = load_bundle()
pipe = bundle["pipeline"]

st.caption(
    f"Model: {bundle['model_name']} | "
    f"Test ROC-AUC: {bundle['test_auc']:.3f}"
)

threshold = st.slider(
    "Spam threshold",
    0.10,
    0.90,
    0.50,
    0.05,
    help=(
        "A message is marked spam if its spam probability is at or above "
        "this value. Lower values catch more spam but may flag real messages."
    )
)

examples = {
    "(type your own)": "",
    "Prize message": (
        "Congratulations! You have won a free gift voucher. "
        "Reply WIN now to claim"
    ),
    "Normal message": "Hey, are we still meeting for lunch tomorrow?",
}

choice = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area("Message", value=examples[choice], height=140)

if st.button("Check message"):
    if not text.strip():
        st.warning("Please enter a message first.")
    else:
        proba = float(pipe.predict_proba([text])[0][1])

        if proba >= threshold:
            st.error(f"Likely SPAM (spam probability {proba:.1%})")
        else:
            st.success(f"Looks OK (spam probability {proba:.1%})")

        st.progress(proba)

        # For linear models, show words that pushed the score towards spam
        model = pipe.named_steps["model"]

        if hasattr(model, "coef_"):
            vec = pipe.named_steps["tfidf"]
            x = vec.transform([text])
            names = vec.get_feature_names_out()

            contrib = x.multiply(model.coef_[0]).tocoo()

            rows = sorted(
                zip(contrib.col, contrib.data),
                key=lambda t: -t[1]
            )[:5]

            rows = [
                (names[c], v)
                for c, v in rows
                if v > 0
            ]

            if rows:
                st.markdown("**Words that pushed towards spam:**")
                st.table(
                    pd.DataFrame(
                        rows,
                        columns=["Word", "Contribution"]
                    ).round(3)
                )
