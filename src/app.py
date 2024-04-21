import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the dataset (assuming it's in the same directory as your Streamlit app)
total_data = pd.read_csv("../data/raw/House_Rent_Dataset.csv")

# Extract unique city names from the 'City' column
unique_cities = total_data['City'].unique()

# In your Streamlit app, or wherever you're making new predictions
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')


# Define a function to get user input
def get_user_input():
    city = st.sidebar.selectbox("City", options=unique_cities)
    size = st.sidebar.number_input("Size", value=100.0, step=10.0, format="%.2f")
    bathroom = st.sidebar.number_input("Number of Bathrooms", value=1, step=1)
    return pd.DataFrame([[city, size, bathroom]], 
                         columns=['City', 'Size', 'Bathroom'])


# Set up the main app
st.title('House Rent Prediction App')
st.write('Enter the details of the house to predict the rent')


# Get user input
user_input = get_user_input()


# Display the user input
st.subheader('Your Input:')
st.write(user_input)


# Predict and display the output
if st.button('Predict'):
   #prediction = model.predict(user_input)
   new_data_transformed = preprocessor.transform(user_input)
   prediction = model.predict(new_data_transformed)
   st.subheader('Predicted Monthly Rent')
   st.write(f"${prediction[0]:,.2f}")