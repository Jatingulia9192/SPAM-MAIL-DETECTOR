import streamlit as st
import pickle
import re

# Load the trained model
with open('trained_spam_model1.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text cleaner to replicate VS Code behavior
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Streamlit UI
st.set_page_config(page_title="Spam Mail Detector", layout="centered")
st.title("📩 Spam Mail Detector")
st.markdown("Enter your email content below to check if it's SPAM or NOT SPAM.")

# Text input
input_mail = st.text_area("✉ Paste the email content here:")

# Prediction
if st.button("Check Now"):
    if input_mail.strip() == "":
        st.warning("⚠ Please enter some email content.")
    else:
        try:
            cleaned_input = clean_text(input_mail)

            # ✅ Show what's being predicted
            st.write("🔍 Cleaned Input:", cleaned_input)

            # Predict
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]
            proba = model.predict_proba(transformed_input)[0]

            st.write(f"🧠 Prediction Confidence — Not Spam: {proba[0]:.2f}, Spam: {proba[1]:.2f}")

            if prediction == 1:
                st.error("🚨 This is SPAM!")
            else:
                st.success("✅ This is NOT SPAM.")
        except Exception as e:
            st.error("❌ Prediction failed.")
            st.exception(e)
