import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('loan_approval_dataset.csv')

# Data preprocessing
df.drop(columns=['loan_id', ' no_of_dependents', ' self_employed', ' income_annum', 
                 ' loan_term', ' residential_assets_value', ' commercial_assets_value', 
                 ' luxury_assets_value', ' bank_asset_value'], inplace=True, axis=1)

X = df[[' cibil_score', ' loan_amount']]
y = df[' loan_status']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scalingc
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Streamlit web app
st.title('Loan Approval Prediction using CIBIL SCORE')

# Input fields for user
cibil_score = st.number_input('Enter CIBIL Score:')
loan_amount = st.number_input('Enter Loan Amount:')

# Prediction and display result
if st.button('Check Loan Approval'):
    new_applicant = [[cibil_score, loan_amount]]
    prediction = clf.predict(new_applicant)

    if prediction[0] == ' Approved':
        if loan_amount<1000:
            st.error("Loan Amount should be greater than 1000")
        else:
            st.success("Congratulations! Your loan has been approved.")

    elif prediction[0] == ' Rejected' and 450<=cibil_score<550 :
        st.error(f"Sorry your Cibil score is {cibil_score} which is less than requirement")
        st.success("You may get Loan with 10% interest")
    elif prediction[0] == ' Rejected' and 350<=cibil_score<450:
        st.error(f"Sorry your Cibil score is {cibil_score} which is less than requirement")
        st.success(f"You may get Loan with 15% interest or ")
        st.(f"You have to increase cibilscore more {500-cibil_score} ")
        
    else:
        st.error("Sorry, your loan has been rejected.")
        st.success(f"You have to increase cibilscore more {int(500-cibil_score)} ")

