# 🏦 Loan Default Predictor

A machine learning web app that predicts whether a loan applicant 
will default based on financial and personal attributes.

## 🔗 Live Demo
[Click here to try the app](https://loan-default-predictor-hqhcpnqzayxtrcksamtoz5.streamlit.app/)

## 📌 Problem Statement
Financial institutions lose billions due to loan defaults. 
This app helps predict default risk early using applicant data,
enabling better lending decisions.

## 📊 Model Performance
| Model | Accuracy |
|---|---|
| Random Forest | XX% |
| Logistic Regression (baseline) | XX% |

## 🛠 Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | ML model training |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Web app deployment |

## 📁 Dataset
- Source: Kaggle — Loan Default Dataset
- Size: 148,670 records
- Features: 34 columns including Credit Score, LTV, Income, 
  Loan Amount, Interest Rate and more

## 🔍 Project Structure
loan-default-predictor/
├── notebooks/
│   └── analysis.ipynb    # EDA + Model Training
├── app.py                # Streamlit web app
├── model.pkl             # Trained Random Forest model
├── features.pkl          # Feature names for prediction
└── requirements.txt      # Dependencies

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

## 💡 Key Findings
- Credit Score is the strongest predictor of default risk
- Higher LTV ratio correlates with increased default probability
- Income alone is not sufficient — debt-to-income ratio matters more

## 👤 Author
Amuruth R
[LinkedIn](your-linkedin-link) | [GitHub](your-github-link)
