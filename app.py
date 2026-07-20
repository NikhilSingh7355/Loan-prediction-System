import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Loan", page_icon="😍")

st.title("🏡 Loan Prediction System")

df = pd.read_csv("loan.csv")

st.write(df)

x = df[["Income", "CIBIL_Score", "Loan_Amount", "Employment_Years"]]
y = df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42
)

model.fit(x_train, y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

income = st.sidebar.number_input(
    "Enter your Income:",
    min_value=1000,
    max_value=1000000,
    value=50000,
    step=1000
)

CIBIL_Score = st.sidebar.number_input(
    "Enter your CIBIL Score:",
    min_value=300,
    max_value=900,
    value=700,
    step=1
)

Loan_Amount = st.sidebar.number_input(
    "Enter your Loan Amount:",
    min_value=5000,
    max_value=1000000,
    value=100000,
    step=1000
)

Employment_Years = st.sidebar.number_input(
    "Enter your Employment Years:",
    min_value=0,
    max_value=50,
    value=5,
    step=1
)

if st.button("Predict"):

    result = model.predict([[income, CIBIL_Score, Loan_Amount, Employment_Years]])

    if result[0] == 1:

        approved = min(
            Loan_Amount,
            int(income * 8 + (CIBIL_Score - 650) * 400)
        )

        st.success("✅ Loan Approved")
        st.success(f"Approved Amount: ₹{approved:,}")

    else:

        st.error("❌ Loan Rejected")