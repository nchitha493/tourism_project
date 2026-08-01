
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction")

st.write(
    "Predict whether a customer is likely to purchase the Wellness Tourism Package."
)

st.write("Enter the customer details below.")

# Customer Details

Age = st.number_input("Age", 18, 100, 30)

CityTier = st.selectbox("City Tier", [1, 2, 3])

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

Passport = st.selectbox(
    "Passport",
    ["Yes", "No"]
)

OwnCar = st.selectbox(
    "Own Car",
    ["Yes", "No"]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=50000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    value=2
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    value=20
)

# Create DataFrame

input_data = pd.DataFrame([{

    "Age": Age,
    "CityTier": CityTier,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched

}])

classification_threshold = 0.45

if st.button("Predict"):

    probability = model.predict_proba(input_data)[0,1]

    prediction = (probability >= classification_threshold).astype(int)

    if prediction == 1:

        st.success(
            f"Customer is likely to purchase the Wellness Tourism Package.\n\nProbability: {probability:.2%}"
        )

    else:

        st.error(
            f"Customer is unlikely to purchase the Wellness Tourism Package.\n\nProbability: {probability:.2%}"
        )
