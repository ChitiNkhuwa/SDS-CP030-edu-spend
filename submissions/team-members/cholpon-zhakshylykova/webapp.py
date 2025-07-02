import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title='International Education Budget Planner', layout='wide')
st.title("International Education Budget Planner")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data_tca_clusters_raw.csv")

data = load_data()

# --- Load Pipeline ---
@st.cache_resource
def load_model_pipeline():
    path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
    return joblib.load(path)

pipeline = load_model_pipeline()
preprocessor = pipeline.named_steps["preprocessor"]
regressor = pipeline.named_steps["regressor"]
encoder = preprocessor.named_transformers_["cat"]
scaler = preprocessor.named_transformers_["num"]

# --- Sidebar Inputs ---
st.sidebar.header("📥 Input Parameters")
target_country = st.sidebar.selectbox("Select Country", sorted(data["Country"].dropna().unique()))
level = st.sidebar.selectbox("Select Level", sorted(data["Level"].dropna().unique()))
duration = st.sidebar.slider("Select Duration (Years)", min_value=1, max_value=6, value=4)

filtered = data[data["Country"] == target_country]
most_common_city = filtered["City"].mode()[0] if not filtered.empty else "Unknown"
most_common_university = filtered["University"].mode()[0] if not filtered.empty else "Unknown"
most_common_program = data["Program"].mode()[0] if not data.empty else "Unknown"

# --- User Numeric Inputs ---
#tuition = st.sidebar.number_input("Tuition per Year (USD)", min_value=0, value=10000)
#living_index = st.sidebar.number_input("Living Cost Index", min_value=0.0, value=65.0)
rent = st.sidebar.number_input("Monthly Rent (USD)", min_value=0, value=500)
visa_fee = st.sidebar.number_input("Visa Fee (USD)", min_value=0, value=200)
insurance = st.sidebar.number_input("Insurance per Year (USD)", min_value=0, value=600)

# --- Construct Input Frame ---
user_input = pd.DataFrame({
    "University": [most_common_university],
    "Program": [most_common_program],
    "Level": [level],
    "Rent_USD": [rent * 12],  # annual
    "Visa_Fee_USD": [visa_fee],
    "Insurance_USD": [insurance],
    "Duration_Years": [duration]
})

# --- Prediction ---
try:
    predicted_tca = pipeline.predict(user_input)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.sidebar.markdown("### 💰 Predicted TCA")
st.sidebar.metric(label="Estimated Total Cost (USD)", value=f"${predicted_tca:,.2f}")

# --- Affordability Map ---
st.subheader("🌍 Global Affordability Map")

fig = px.choropleth(
    data_frame=data,
    locations="Country",
    locationmode="country names",
    color="Total_cost",
    hover_name="Country",
    color_continuous_scale="Turbo",
    range_color=(data["Total_cost"].min(), data["Total_cost"].max()),
    title="🌍 Average Total Cost of Attendance by Country",
    labels={"Total_cost": "Total Cost (USD)"},
)

fig.update_geos(
    showframe=False,
    showcoastlines=False,
    projection_type="natural earth",
)

fig.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    geo_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=14),
    title_font=dict(size=20, color="white"),
    coloraxis_colorbar=dict(
        title="Total Cost (USD)",
        tickprefix="$",
        thickness=15,
        len=0.75,
        bgcolor="rgba(0,0,0,0)",
        title_side="right",
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- Cluster Explorer ---
st.subheader("📊 Cost Cluster Explorer")
cluster_cols = [col for col in data.columns if col.endswith("_Cluster")]
if cluster_cols:
    cluster_option = st.selectbox("Select Cluster Type", cluster_cols)
    cluster_summary = data.groupby(cluster_option).mean(numeric_only=True).reset_index()
    st.dataframe(cluster_summary)

    with st.expander("ℹ️ Cluster Segment Descriptions"):
        st.markdown("""
        <ul>
        <li><b>Low Cost Cluster</b>: Total annual costs typically below <b>$20,000</b>.</li>
        <li><b>Medium Cost Cluster</b>: Between <b>$20,000</b> and <b>$40,000</b>.</li>
        <li><b>High Cost Cluster</b>: Above <b>$40,000</b> annually.</li>
        </ul>
        """, unsafe_allow_html=True)
else:
    st.warning("No cluster columns found in the dataset.")

# --- Instructions ---
st.markdown("""
---
### 📘 How to Use This App

1. Select your **target country**, **education level**, and **study duration**.
2. Enter estimated costs like **tuition**, **rent**, **insurance**, etc.
3. View your **Total Cost of Attendance** prediction.
4. Explore cost trends with:
   - 🌍 Affordability Map
   - 📊 Cost Cluster Explorer
5. Use the cluster guide to compare regions or institutions.
""")
