import pickle
import numpy as np
import streamlit as st

# Load the saved model from the file
filename = 'final_model1.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Custom CSS for colorful representation
st.markdown(
    """
    <style>
    .title {
        color: #FF5733;
        text-align: center;
        font-size: 32px;
    }
    .text {
        color: #7D3C98;
        text-align: center;
        font-size: 18px;
    }
    .prediction {
        color: #6C3483;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .footer {
        color: #999999;
        text-align: center;
        font-size: 14px;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the Streamlit web app
st.markdown('<p class="title">Weight Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="text">Enter your height in feet to predict your weight.</p>', unsafe_allow_html=True)

# Default value for height
default_height = 5.8

# Input height from the user with limits and step size
height_input = st.number_input(
    "Enter the height in feet:",
    value=default_height,
    min_value=1.0,
    max_value=8.0,
    step=0.1
)

# Predict button
if st.button('Predict'):
    try:
        # Reshape the input height to match the shape expected by the model (2D array)
        height_input_2d = np.array(height_input).reshape(1, -1)

        # Use the loaded model to make predictions
        predicted_weight = loaded_model.predict(height_input_2d)
        predicted_value = predicted_weight[0] if predicted_weight.ndim == 1 else predicted_weight[0, 0]

        # Print the predicted weight
        st.markdown(f'<p class="prediction">Predicted weight: {predicted_value:.2f} kg</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown('<hr><p class="footer">Made with ❤️ using Streamlit</p>', unsafe_allow_html=True)
