import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.joblib')

def predict(features):
    return model.predict(features)

st.title('Machine Learning Model Interface')

st.write('Enter the features to get a prediction:')

# Create input fields for features
feature1 = st.number_input('Feature 1', value=0.0)
feature2 = st.number_input('Feature 2', value=0.0)
feature3 = st.number_input('Feature 3', value=0.0)

# Create a button to trigger prediction
if st.button('Predict'):
    features = pd.DataFrame([[feature1, feature2, feature3]], 
                            columns=['feature1', 'feature2', 'feature3'])
    prediction = predict(features)
    st.write(f'The prediction is: {prediction[0]}')

st.write('Note: This is a basic interface. Adjust the features and model loading as per your specific ML model.')