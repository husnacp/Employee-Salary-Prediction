
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

with st.sidebar:
    st.header("📥 Input Employee Details")

    # Input Fields
    age = st.slider("Age", 18, 75, 30)
    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Others'
    ])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 50000)
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ])
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces', 'Others'
    ])
    relationship = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family',
        'Other-relative', 'Unmarried'
    ])
    race = st.selectbox("Race", [
        'White', 'Black', 'Asian-Pac-Islander',
        'Amer-Indian-Eskimo', 'Other'
    ])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany',
        'Canada', 'India', 'Others'
    ])
    educational_num = st.slider("Education Num", 1, 16, 10)

# Label Encodings matching your training pipeline
label_encoders = {
    'workclass': {'Private': 3, 'Self-emp-not-inc': 5, 'Self-emp-inc': 4, 'Federal-gov': 0, 'Local-gov': 1, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 2, 'Others': 8},
    'marital-status': {'Married-civ-spouse': 1, 'Divorced': 0, 'Never-married': 2, 'Separated': 3, 'Widowed': 5, 'Married-spouse-absent': 4, 'Married-AF-spouse': 6},
    'occupation': {'Tech-support': 12, 'Craft-repair': 3, 'Other-service': 8, 'Sales': 10, 'Exec-managerial': 4, 'Prof-specialty': 9,
                   'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 2, 'Transport-moving': 13,
                   'Priv-house-serv': 11, 'Protective-serv': 7, 'Armed-Forces': 1, 'Others': 14},
    'relationship': {'Wife': 5, 'Own-child': 3, 'Husband': 2, 'Not-in-family': 0, 'Other-relative': 1, 'Unmarried': 4},
    'race': {'White': 4, 'Black': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3},
    'gender': {'Male': 1, 'Female': 0},
    'native-country': {'United-States': 38, 'Mexico': 20, 'Philippines': 29, 'Germany': 11, 'Canada': 4, 'India': 15, 'Others': 0}
}

# Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [label_encoders['workclass'].get(workclass, 8)],
    'fnlwgt': [fnlwgt],
    'marital-status': [label_encoders['marital-status'].get(marital_status, 0)],
    'occupation': [label_encoders['occupation'].get(occupation, 14)],
    'relationship': [label_encoders['relationship'].get(relationship, 0)],
    'race': [label_encoders['race'].get(race, 4)],
    'gender': [label_encoders['gender'][gender]],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [label_encoders['native-country'].get(native_country, 0)],
    'educational-num': [educational_num]
})

# Reorder columns to match training
input_df = input_df[model.feature_names_in_]

# Display the input
st.subheader("🔎 Input Data Preview")
st.write(input_df)

# Bar chart for numeric fields
st.subheader("📊 Input Feature Distribution")
fig, ax = plt.subplots()
input_df[['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'educational-num']].T.plot(kind='barh', legend=False, ax=ax)
st.pyplot(fig)

# Prediction
if st.button("🔮 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.markdown("### 🧾 Prediction Result")
    st.success(f"🎯 **Prediction**: {'>50K' if prediction == 1 else '<=50K'}")
    
    # Pie chart for probability
    st.subheader("📈 Prediction Confidence")
    fig2, ax2 = plt.subplots()
    ax2.pie(proba, labels=['<=50K', '>50K'], autopct='%1.1f%%', startangle=90, colors=["#ff9999", "#66b3ff"])
    ax2.axis('equal')
    st.pyplot(fig2)
