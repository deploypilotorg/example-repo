import streamlit as st
import pandas as pd
import numpy as np

st.title('Machine Learning Model Interface Demo')

st.write('Enter the features to get a mock prediction:')

# Create input fields for features
feature1 = st.number_input('Feature 1', value=0.0)
feature2 = st.number_input('Feature 2', value=0.0)
feature3 = st.number_input('Feature 3', value=0.0)

# Create a button to trigger prediction
if st.button('Predict'):
    # Mock prediction using a simple function
    prediction = feature1 + feature2 * 2 + feature3 * 3
    st.write(f'The mock prediction is: {prediction:.2f}')
    
    # Display a demo visualization
    st.write('Feature importance (demo):')
    feature_importance = pd.DataFrame({
        'Feature': ['Feature 1', 'Feature 2', 'Feature 3'],
        'Importance': [1, 2, 3]
    })
    st.bar_chart(feature_importance.set_index('Feature'))

st.write('Note: This is a demo interface without an actual ML model.')
