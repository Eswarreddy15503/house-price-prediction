import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("🏠 House Price Prediction App")

# Load dataset
data = pd.read_csv("data.csv")

# Select columns
data = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'price']]
data = data.dropna()

# Features and target
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Enter House Details")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)
sqft = st.sidebar.number_input("Sqft Living", 500, 5000, 2000)
floors = st.sidebar.number_input("Floors", 1, 3, 1)

# Prediction
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft, floors]],
                              columns=['bedrooms', 'bathrooms', 'sqft_living', 'floors'])

    prediction = model.predict(input_data)[0]

    st.subheader("💰 Predicted House Price:")
    st.success(f"${round(prediction, 2)}")

# ------------------ GRAPHS ------------------

st.subheader("📊 Data Visualization")

# 1. Area vs Price
fig1 = plt.figure()
plt.scatter(data['sqft_living'], data['price'])
plt.xlabel("Sqft Living")
plt.ylabel("Price")
plt.title("Area vs Price")
st.pyplot(fig1)

# 2. Bedrooms vs Price
fig2 = plt.figure()
plt.scatter(data['bedrooms'], data['price'])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs Price")
st.pyplot(fig2)

# 3. Actual vs Predicted
predictions = model.predict(X)

fig3 = plt.figure()
plt.scatter(y, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
st.pyplot(fig3)

# Dataset preview
st.subheader("📄 Dataset Preview")
st.write(data.head())