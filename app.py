import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and encoders
rf_model = joblib.load('rf_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the input form
st.title("Credit Card Fraud Detection")

st.header("Enter Transaction Details:")
cc_num = st.text_input("Credit Card Number", "1234567890")
merchant = st.text_input("Merchant", "Store A")
category = st.selectbox("Category", ["Electronics", "Grocery", "Clothing", "Other"])
amt = st.number_input("Transaction Amount", min_value=0.01, value=100.50)
first = st.text_input("First Name", "John")
last = st.text_input("Last Name", "Doe")
gender = st.selectbox("Gender", ["M", "F"])
street = st.text_input("Street", "123 Main St")
city = st.text_input("City", "City A")
state = st.text_input("State", "State A")
zip_code = st.number_input("ZIP Code", min_value=10000, value=12345)
lat = st.number_input("Latitude", value=40.7128)
long = st.number_input("Longitude", value=-74.0060)
city_pop = st.number_input("City Population", value=1000000)
job = st.text_input("Job", "Engineer")
trans_num = st.text_input("Transaction Number", "987654321")
unix_time = st.number_input("Unix Time", min_value=1, value=1622518243)
merch_lat = st.number_input("Merchant Latitude", value=40.7100)
merch_long = st.number_input("Merchant Longitude", value=-74.0050)
year = st.number_input("Transaction Year", min_value=2000, max_value=2024, value=2021)
month = st.number_input("Transaction Month", min_value=1, max_value=12, value=5)
day = st.number_input("Transaction Day", min_value=1, max_value=31, value=25)
hour = st.number_input("Transaction Hour", min_value=0, max_value=23, value=14)
age = st.number_input("Age", min_value=18, value=30)

# Process the input
input_data = pd.DataFrame({
    'cc_num': [cc_num],
    'merchant': [merchant],
    'category': [category],
    'amt': [amt],
    'first': [first],
    'last': [last],
    'gender': [gender],
    'street': [street],
    'city': [city],
    'state': [state],
    'zip': [zip_code],
    'lat': [lat],
    'long': [long],
    'city_pop': [city_pop],
    'job': [job],
    'trans_num': [trans_num],
    'unix_time': [unix_time],
    'merch_lat': [merch_lat],
    'merch_long': [merch_long],
    'year': [year],
    'month': [month],
    'day': [day],
    'hour': [hour],
    'age': [age]
})

# Encode categorical columns and handle unseen labels
for col in label_encoders:
    if col in input_data.columns:
        known_classes = label_encoders[col].classes_
        input_data[col] = input_data[col].apply(lambda x: x if x in known_classes else known_classes[0])
        input_data[col] = label_encoders[col].transform(input_data[col])

# Ensure columns match the training data
X_columns = rf_model.feature_names_in_
input_data = input_data.reindex(columns=X_columns, fill_value=0)

# Make the prediction
if st.button("Predict"):
    prediction = rf_model.predict(input_data)
    result = "Fraudulent Transaction" if prediction[0] == 1 else "Genuine Transaction"
    st.subheader(f"Prediction: {result}")
