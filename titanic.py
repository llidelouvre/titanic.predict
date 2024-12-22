import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set background color to blue-white and adjust text/button colors using custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #E3F2FD;  /* Light blue background */
            color: #0D47A1;  /* Dark blue text */
        }
        .stButton>button {
            background-color: #0D47A1;  /* Dark blue button background */
            color: white;  /* White text for button */
        }
        .stSidebar {
            background-color: #E3F2FD;  /* Light blue sidebar background */
        }
        .stTextInput>div>input {
            color: #0D47A1;  /* Dark blue text input color */
        }
        .stSlider>div>div>div>input {
            color: #0D47A1;  /* Dark blue color for sliders */
        }
        .stSelectbox>div>div>input {
            color: #0D47A1;  /* Dark blue color for select boxes */
        }
        .centered-text {
            text-align: center;  /* Center-align text */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
with open('model_regresi.pkl', 'rb') as file:
    model = pickle.load(file)

st.markdown('<h2 class="centered-text">WELCOME TO</h2>', unsafe_allow_html=True)
st.markdown('<h1 class="centered-text">🚢 TITANIC SURVIVAL PREDICTION</h1>', unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #0D47A1;'>", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.markdown('<h3 class="centered-text">📝 INPUT PASSENGER DATA</h3>', unsafe_allow_html=True)

nama = st.sidebar.text_input('👤 Passenger Name', placeholder="Enter passenger name here")
t_number = st.sidebar.text_input('🎟️ Ticket Number', placeholder="Enter ticket number here")

def user_input_features():
    sex = st.sidebar.selectbox('⚧️ Gender', ["Select Gender", 'Female', 'Male'], index=0) 
    age = st.sidebar.number_input('🎂 Age', min_value=0, max_value=100, value=0)
    sibsp = st.sidebar.number_input('👥 No of Siblings or Spouses on Board', min_value=0, max_value=100, value=0)
    parch = st.sidebar.number_input('👪 No of Parents or Children on Board', min_value=0, max_value=100, value=0)
    st.sidebar.markdown(
        """
            <div style="background-color: #0D47A1; padding: 10px; border-radius: 5px;">
                <strong style="color: white;">💡 Passenger Class Info</strong><br>
                <span style="color: white;">Class 1 => 75.0 to 500.0</span><br>
                <span style="color: white;">Class 2 => 70.0 to 75.0</span><br>
                <span style="color: white;">Class 3 => 0.0 to 70.0</span>
            </div>
        """,
        unsafe_allow_html=True
    )
    pclass = st.sidebar.selectbox('🏷️ Passenger Class', ["Select Class", 1, 2, 3], index=0)
    if pclass == 1:
        fare = st.sidebar.number_input('💰 Passenger Fare', min_value=75.0, max_value=500.0, value=75.0)
    elif pclass == 2:
        fare = st.sidebar.number_input('💰 Passenger Fare', min_value=70.0, max_value=75.0, value=70.0)
    else:
        fare = st.sidebar.number_input('💰 Passenger Fare', min_value=0.0, max_value=70.0, value=0.0)
    cabin = st.sidebar.text_input('🚪 Cabin', placeholder="Enter cabin here")
    embarked = st.sidebar.selectbox('⛴️ Port of Embarkation', ["Select Embarkation", 'Cherbourg', 'Queenstown', 'Southampton'], index=0)

    # Encode embarked
    if embarked == "Select Embarkation":
        embarked_encoded = None
    else:
        embarked_encoded = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}[embarked]

    # Encode gender
    if sex == "Select Gender":
        sex_encoded = None
    else:
        sex_encoded = {'female': 0, 'male': 1}[sex.lower()]

    # Create dataframe
    data = {
        'Sex': sex_encoded,
        'Age': age,
        'No of Siblings or Spouses on Board': sibsp,
        'No of Parents or Children on Board': parch,
        'Passenger Class': pclass,
        'Passenger Fare': fare,
        'Cabin': cabin,
        'Port of Embarkation': embarked_encoded,
    }

    return pd.DataFrame(data, index=[0])

# Input pengguna
input_df = user_input_features()

# Button for prediction
if st.sidebar.button('Prediction'):

    missing_inputs = []
    if not nama:
        missing_inputs.append("Name")
    if not t_number:
        missing_inputs.append("Ticket Number")
    if input_df['Passenger Class'].iloc[0] == "Select Class":
        missing_inputs.append("Passenger Class")
    if input_df['Sex'].iloc[0] is None:
        missing_inputs.append("Gender")
    if pd.isnull(input_df['Cabin'].iloc[0]) or input_df['Cabin'].iloc[0] == "":
        missing_inputs.append("Cabin")
    if input_df['Port of Embarkation'].iloc[0] is None:
        missing_inputs.append("Port of Embarkation")

    # If any inputs are missing
    if missing_inputs:
        st.error(f"**Please fill in all required fields: {', '.join(missing_inputs)}**")
    else:
        # Make prediction using the model
        prediction = model.predict(input_df)
        prediction_text = 'Survived' if prediction[0] == 1 else 'Not Survived'

        # Display the result
        st.markdown(f"#### Passenger Name: {nama}", unsafe_allow_html=True)

        if prediction[0] == 0:
            st.markdown(f"##### Prediction Result: I'm sorry, it looks like you did not survive 😞💔", unsafe_allow_html=True)
        else:
            st.markdown(f"##### Prediction Result: Congratulations, it looks like you will survive 😊🎉", unsafe_allow_html=True)

        # Plot the prediction result
        fig, ax = plt.subplots()

        # Set plot background color to blue
        fig.patch.set_facecolor('#E3F2FD')

        # Bar plot
        ax.bar(['Survived', 'Not Survived'], [prediction[0], 1 - prediction[0]], color=['green', 'red'])

        # Set plot text color
        ax.set_title('Passenger Safety Predictions', color='#0D47A1')
        ax.set_ylabel('Probability', color='#0D47A1')
        ax.tick_params(axis='x', colors='#0D47A1')
        ax.tick_params(axis='y', colors='#0D47A1')

        st.pyplot(fig)