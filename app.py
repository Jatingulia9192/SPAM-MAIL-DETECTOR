
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

st.title("Spam Message Detector")
st.caption("Check whether an SMS or email is spam.")

@st.cache_resource
def load_sms_model():
    return joblib.load("models/spam_model.joblib")


@st.cache_resource
def load_email_model():
    return joblib.load("models/email_spam_model.joblib")


choice = st.selectbox(
    "Select what you want to check",
    ["SMS Message", "Email"]
)

if choice == "SMS Message":
    bundle = load_sms_model()
else:
    bundle = load_email_model()

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
    help="Lower values catch more spam but may flag real messages."
)

if choice == "SMS Message":
    examples = {
        "(type your own)": "",
        "Prize message": (
            "Congratulations! You have won a free gift voucher. "
            "Reply WIN now to claim"
        ),
        "Normal message": "Hey, are we still meeting for lunch tomorrow?"
    }
else:
    examples = {
        "(type your own)": "",
        "Spam email": (
            "Subject: You have won a prize! "
            "Click here to claim your free cash reward."
        ),
        "Normal email": (
            "Subject: Meeting tomorrow "
            "Hi, please find the meeting details attached. "
            "Let me know if you have any questions."
        )
    }

example = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area(
    "Enter your message or email",
    value=examples[example],
    height=180
)

if st.button("Check for Spam"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        proba = float(pipe.predict_proba([text])[0][1])

        if proba >= threshold:
            st.error(f"Likely SPAM (spam probability {proba:.1%})")
        else:
            st.success(f"Looks OK (spam probability {proba:.1%})")

        st.progress(proba)

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

