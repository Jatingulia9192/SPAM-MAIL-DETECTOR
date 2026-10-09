# SMS and Email Spam Detector

This is a machine learning project that checks whether a message or email is spam or normal. I built a Streamlit web application where users can enter a message or email and get a prediction.

The project uses TF-IDF to convert text into numerical features and machine learning models to classify the text. I trained separate models for SMS messages and emails.

## Features

- Detects spam in SMS messages
- Detects spam in emails
- Shows the spam probability for the entered text
- Allows users to try sample messages and emails
- Shows words that pushed the prediction towards spam
- Provides a simple interface using Streamlit

## Datasets

### SMS Dataset

The SMS model uses the UCI SMS Spam Collection dataset. It contains messages labelled as:

- **Ham:** Normal messages
- **Spam:** Unwanted or spam messages

### Email Dataset

The email model uses the Apache SpamAssassin Public Corpus. I used the `easy_ham` and `spam` folders to prepare the email dataset for training.

The email dataset contains both normal emails and spam emails. The email subject and text content are used by the model for prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- TF-IDF

## Project Work

In this project, I worked on the following steps:

- Loaded and cleaned the SMS dataset
- Prepared the email dataset from raw email files
- Removed duplicate messages and emails
- Converted text into numerical features using TF-IDF
- Compared different machine learning models for SMS classification
- Trained a separate model for email spam detection
- Evaluated the models using precision, recall, F1-score and ROC-AUC
- Saved the trained models using Joblib
- Created a Streamlit application for testing SMS messages and emails

## Models Used

For the SMS classifier, I compared the following models:

- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression
- Random Forest

For the email classifier, I used Logistic Regression with TF-IDF.

## Email Model Results

The email model was tested on a held-out test set containing 592 emails.

- **Accuracy:** 99%
- **Spam precision:** 99%
- **Spam recall:** 94%
- **Spam F1-score:** 96%
- **ROC-AUC:** 0.9984

These results are based on the test split used during training. Actual performance may be different on new emails.

## Project Structure

```text
SPAM-MAIL-DETECTOR/
|
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── setup.sh
|
├── assets/
|
├── data/
|   ├── SMSSpamCollection
|   └── email_raw/
|       ├── easy_ham/
|       └── spam/
|
├── models/
|   ├── spam_model.joblib
|   └── email_spam_model.joblib
|
├── notebooks/
|   └── 01_spam.ipynb
|
└── src/
    ├── train_utils.py
    └── train_email_model.py
```

## How to Run the Project

First, clone the repository:

```bash
git clone https://github.com/Jatingulia9192/SPAM-MAIL-DETECTOR.git
```

Go to the project folder:

```bash
cd SPAM-MAIL-DETECTOR
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser. Select SMS Message or Email, enter the text, and click the check button to get a prediction.

## Limitations

The models are trained on specific datasets, so they may not correctly identify every spam message or email.

The email dataset is relatively old and mainly contains English emails. New spam messages may use different words and patterns, which can affect the predictions.

The spam probability shown by the application is a model prediction and does not guarantee that a message is spam.

## Author

Jatin Gulia

B.Tech Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning  
DIT University, Dehradun