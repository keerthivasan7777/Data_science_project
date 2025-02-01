import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Load the dataset
dataset = pd.read_csv('C:\\Users\\Keerthivasan R\\OneDrive\\Desktop\\Internship\\Advertising.csv')

dataset["total_spend"] = dataset["TV"] + dataset["radio"] + dataset["newspaper"]
X = dataset[["TV", "radio", "newspaper"]].values
y = dataset["sales"].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Streamlit App
st.title("Advertising Sales Prediction")
st.write("Predict sales based on advertising spend in TV, Radio, and Newspaper.")

# User input for TV, Radio, and Newspaper spending
tv_spend = st.number_input("Enter TV Advertising Spend ($)", min_value=0.0,format="%.2f")
radio_spend = st.number_input("Enter Radio Advertising Spend ($)", min_value=0.0, format="%.2f")
newspaper_spend = st.number_input("Enter Newspaper Advertising Spend ($)", min_value=0.0, format="%.2f")

if st.button("Predict Sales"):
    input_data = np.array([[tv_spend, radio_spend, newspaper_spend]])
    prediction = regressor.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f} units")


