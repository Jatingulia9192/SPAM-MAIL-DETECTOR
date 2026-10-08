# SMS Spam Classifier

This project is a machine learning based SMS spam classifier. It takes a text message as input and predicts whether the message is spam or a normal message.

The project uses TF-IDF for converting text into numerical features and different machine learning algorithms for classification. A Streamlit application is also created so that the model can be tested through a simple web interface.

## Dataset

The dataset used in this project is the UCI SMS Spam Collection.

It contains SMS messages classified into two categories:

- ham - normal message
- spam - spam message

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- TF-IDF

## Project Work

In this project, I performed the following steps:

- Loaded and cleaned the SMS dataset
- Removed duplicate messages
- Prepared the data for machine learning
- Used TF-IDF to convert text into numerical features
- Compared different machine learning models
- Evaluated the models using precision, recall, F1-score and ROC-AUC
- Selected the best model based on the results
- Created a pipeline containing the TF-IDF vectorizer and machine learning model
- Saved the trained model using Joblib
- Developed a Streamlit application for testing new messages

## Models Used

The following models were compared:

- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression
- Random Forest



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
|   └── SMSSpamCollection
|
├── models/
|   └── spam_model.joblib
|
├── notebooks/
|   └── 01_spam.ipynb
|
└── src/
```

## How to Run the Project

First clone the repository:

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

The application will open in the browser.

## Limitations

The model is trained on SMS messages, so its performance may not be the same for long emails or other types of text.

The dataset is also relatively old and mainly contains English messages. Spam messages can change over time, so the model may not correctly identify every new type of spam message.

## Author

Jatin Gulia

B.Tech Computer Science and Engineering
Specialization: Artificial Intelligence and Machine Learning
DIT University, Dehradun
```
