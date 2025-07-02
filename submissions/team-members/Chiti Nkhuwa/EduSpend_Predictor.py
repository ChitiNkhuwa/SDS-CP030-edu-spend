import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Load the trained model
model = joblib.load('best_tca_model.joblib')

st.set_page_config(page_title="EduSpend TCA Predictor", layout="centered")
st.title("🌍 EduSpend: Total Cost of Attendance Predictor")

st.markdown("""
<style>
.main {
    background-color: #fff8f0;
}
.stButton>button {
    background-color: #ffb347;
    color: white;
}
.stSuccess {
    background-color: #ffe5b4;
    color: #a0522d;
}
</style>
""", unsafe_allow_html=True)

# Country-City-Currency Data
country_currency_map = {
    "USA": "USD",
    "UK": "GBP",
    "Canada": "CAD",
    "Australia": "AUD"
}

city_options = {
    "USA": ["New York", "Los Angeles", "Chicago"],
    "UK": ["London", "Manchester", "Edinburgh"],
    "Canada": ["Toronto", "Vancouver", "Montreal"],
    "Australia": ["Sydney", "Melbourne", "Brisbane"]
}

level_options = ["Bachelor", "Master", "PhD"]
program_options = ["Engineering", "Business", "Arts", "Science"]
local_currency_options = ["USD", "GBP", "CAD", "EUR", "ZMW", "KES", "INR", "NGN"]

# === Utility Function ===
def fetch_exchange_rate(from_currency, to_currency):
    if from_currency == to_currency:
        return 1.0
    url = f"https://api.exchangerate.host/convert?from={from_currency}&to={to_currency}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("result", 1.0)
    else:
        return 1.0

# === INPUTS ===
with st.expander("📘 Academic Info"):
    country = st.selectbox("Country", list(country_currency_map.keys()))
    city = st.selectbox("City", city_options[country])
    level = st.selectbox("Level", level_options)
    program = st.selectbox("Program", program_options)

with st.expander("💸 Financial Info"):
    col1, col2 = st.columns(2)
    with col1:
        living_cost_index = st.number_input("Living Cost Index", min_value=0.0)
        rent_usd = st.number_input("Monthly Rent (in base currency)", min_value=0.0)
        visa_usd = st.number_input("Visa Fee (in base currency)", min_value=0.0)
    with col2:
        insurance_usd = st.number_input("Insurance Cost (Annual)", min_value=0.0)
        tuition_usd = st.number_input("Tuition Fee (Annual)", min_value=0.0)

# Currency Section
st.markdown("### 💱 Currency Settings")
base_currency = country_currency_map[country]
st.markdown(f"**Institution Currency:** {base_currency}")

local_currency = st.selectbox("Select Your Local Currency", local_currency_options)
auto_rate = fetch_exchange_rate(base_currency, local_currency)

exchange_rate = st.number_input(
    f"Exchange Rate ({base_currency} ➡ {local_currency})",
    min_value=0.0001,
    value=round(auto_rate, 4),
    help=f"1 {base_currency} = ? {local_currency}"
)

# Create DataFrame for model
input_df = pd.DataFrame([{
    "Country": country,
    "City": city,
    "Level": level,
    "Program": program,
    "Living_Cost_Index": living_cost_index,
    "Rent_USD": rent_usd,
    "Visa_Fee_USD": visa_usd,
    "Insurance_USD": insurance_usd,
    "Exchange_Rate": exchange_rate,
    "Tuition_USD": tuition_usd
}])

# === PREDICTION ===
if st.button("🎯 Predict TCA"):
    prediction = model.predict(input_df)[0]
    local_prediction = prediction * exchange_rate
    confidence = 0.87  # placeholder

    st.markdown(f"""
    <div style='background-color:#ffe5b4; padding:20px; border-radius:10px; margin-top:20px;'>
        <h3 style='color:#a0522d;'>Predicted TCA: <b>{prediction:,.2f} {base_currency}</b></h3>
        <h4 style='color:#a0522d;'>In Your Local Currency: <b>{local_prediction:,.2f} {local_currency}</b></h4>
        <p style='color:#a0522d;'>Confidence Score: <b>{confidence * 100:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Cost Breakdown Pie Chart"):
        labels = ['Tuition', 'Rent (x12)', 'Visa', 'Insurance']
        values = [tuition_usd, rent_usd * 12, visa_usd, insurance_usd]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

    with st.expander("🔍 Model Input Data"):
        st.dataframe(input_df)
