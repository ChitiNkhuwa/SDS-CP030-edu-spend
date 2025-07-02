import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import joblib

# Page configuration
st.set_page_config(
    page_title="EduSpend Analytics",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/International_Education_Costs.csv')
    
    # Calculate TCA and other features
    df['TCA'] = (df['Tuition_USD'] + 
                 (df['Rent_USD'] * 12) + 
                 df['Visa_Fee_USD'] + 
                 (df['Insurance_USD'] * df['Duration_Years']))
    
    df['Tuition_to_Rent_Ratio'] = df['Tuition_USD'] / df['Rent_USD']
    df['Total_Living_Cost'] = (df['Rent_USD'] * 12 + 
                              df['Insurance_USD'] * df['Duration_Years'])
    df['Cost_per_Year'] = df['TCA'] / df['Duration_Years']
    df['Affordability_Tier'] = pd.qcut(df['TCA'], q=3, 
                                     labels=['Low', 'Medium', 'High'])
    return df

@st.cache_resource
def load_models():
    try:
        tca_model = joblib.load('models/tca_predictor.joblib')
        affordability_model = joblib.load('models/affordability_classifier.joblib')
        return tca_model, affordability_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Load models
tca_model, affordability_model = load_models()
models_loaded = tca_model is not None and affordability_model is not None

if not models_loaded:
    st.warning("Models not loaded successfully. Cost prediction features will be disabled.")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Overview", "Cost Explorer", "Program Comparison", "Cost Predictor"],
        icons=['house', 'graph-up', 'bar-chart', 'calculator'],
        menu_icon="cast",
        default_index=0,
    )

# Overview Page
if selected == "Overview":
    st.title("🎓 Global Education Cost Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Programs", len(df))
        st.metric("Average Total Cost", f"${df['TCA'].mean():,.2f}")
    
    with col2:
        st.metric("Countries Covered", len(df['Country'].unique()))
        st.metric("Average Tuition", f"${df['Tuition_USD'].mean():,.2f}")
    
    # World map of costs
    st.subheader("Global Cost Distribution")
    fig = px.scatter_geo(df,
                        locations='Country',
                        locationmode='country names',
                        size='TCA',
                        color='Affordability_Tier',
                        hover_data=['Program', 'University', 'TCA'],
                        title='Education Costs Worldwide')
    st.plotly_chart(fig, use_container_width=True)

# Cost Explorer Page
elif selected == "Cost Explorer":
    st.title("📊 Cost Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Select Country", ['All'] + sorted(df['Country'].unique()))
    with col2:
        program = st.selectbox("Select Program", ['All'] + sorted(df['Program'].unique()))
    with col3:
        level = st.selectbox("Select Level", ['All'] + sorted(df['Level'].unique()))
    
    # Filter data
    filtered_df = df.copy()
    if country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == country]
    if program != 'All':
        filtered_df = filtered_df[filtered_df['Program'] == program]
    if level != 'All':
        filtered_df = filtered_df[filtered_df['Level'] == level]
    
    # Cost distribution
    st.subheader("Cost Distribution")
    fig = px.box(filtered_df, 
                 y='TCA',
                 color='Level',
                 title='Total Cost Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown
    st.subheader("Cost Breakdown")
    cost_breakdown = pd.DataFrame({
        'Component': ['Tuition', 'Rent', 'Insurance', 'Visa'],
        'Average Cost': [
            filtered_df['Tuition_USD'].mean(),
            filtered_df['Rent_USD'].mean() * 12,
            filtered_df['Insurance_USD'].mean(),
            filtered_df['Visa_Fee_USD'].mean()
        ]
    })
    fig = px.pie(cost_breakdown, 
                 values='Average Cost', 
                 names='Component',
                 title='Average Cost Breakdown')
    st.plotly_chart(fig, use_container_width=True)

# Program Comparison Page
elif selected == "Program Comparison":
    st.title("🔄 Program Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        program1 = st.selectbox("Select First Program", sorted(df['Program'].unique()), key='prog1')
        country1 = st.selectbox("Select First Country", sorted(df['Country'].unique()), key='country1')
    
    with col2:
        program2 = st.selectbox("Select Second Program", sorted(df['Program'].unique()), key='prog2')
        country2 = st.selectbox("Select Second Country", sorted(df['Country'].unique()), key='country2')
    
    # Get program data
    prog1_data = df[(df['Program'] == program1) & (df['Country'] == country1)]
    prog2_data = df[(df['Program'] == program2) & (df['Country'] == country2)]
    
    # Comparison metrics
    metrics = ['TCA', 'Tuition_USD', 'Total_Living_Cost']
    
    # Create comparison chart
    comparison_data = []
    for metric in metrics:
        comparison_data.extend([
            {'Program': f"{program1} ({country1})", 
             'Metric': metric.replace('_', ' '), 
             'Value': prog1_data[metric].mean()},
            {'Program': f"{program2} ({country2})", 
             'Metric': metric.replace('_', ' '), 
             'Value': prog2_data[metric].mean()}
        ])
    
    comparison_df = pd.DataFrame(comparison_data)
    fig = px.bar(comparison_df, 
                 x='Metric', 
                 y='Value', 
                 color='Program',
                 barmode='group',
                 title='Program Cost Comparison')
    st.plotly_chart(fig, use_container_width=True)

# Cost Predictor Page
elif selected == "Cost Predictor":
    st.title("🔮 Cost Predictor")
    
    if not models_loaded:
        st.error("Cost prediction is currently unavailable. Please ensure models are properly loaded.")
    else:
        st.write("Enter program details to predict total cost of attendance:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_country = st.selectbox("Country", sorted(df['Country'].unique()))
            pred_city = st.selectbox("City", sorted(df[df['Country'] == pred_country]['City'].unique()))
            pred_university = st.selectbox("University", sorted(df[df['Country'] == pred_country]['University'].unique()))
            pred_program = st.selectbox("Program", sorted(df['Program'].unique()))
            pred_level = st.selectbox("Level", sorted(df['Level'].unique()))
        
        with col2:
            pred_duration = st.slider("Duration (years)", 1.0, 5.0, 2.0, 0.5)
            pred_tuition = st.slider("Tuition (USD)", 500, 100000, 20000, 500)
        
        if st.button("Predict Cost"):
            # Create input data
            # Get reference values
            rent_usd = df[df['City'] == pred_city]['Rent_USD'].mean()
            insurance_usd = df[df['Country'] == pred_country]['Insurance_USD'].mean()
            visa_fee = df[df['Country'] == pred_country]['Visa_Fee_USD'].mean()
            
            # Calculate derived features
            total_living_cost = (rent_usd * 12 + insurance_usd * pred_duration)
            tuition_to_rent_ratio = pred_tuition / rent_usd if rent_usd > 0 else 0
            cost_per_year = (pred_tuition + total_living_cost + visa_fee) / pred_duration
            
            input_data = pd.DataFrame({
                'Country': [pred_country],
                'City': [pred_city],
                'University': [pred_university],
                'Program': [pred_program],
                'Level': [pred_level],
                'Duration_Years': [pred_duration],
                'Tuition_USD': [pred_tuition],
                'Living_Cost_Index': [df[df['City'] == pred_city]['Living_Cost_Index'].mean()],
                'Rent_USD': [rent_usd],
                'Visa_Fee_USD': [visa_fee],
                'Insurance_USD': [insurance_usd],
                'Exchange_Rate': [df[df['Country'] == pred_country]['Exchange_Rate'].mean()],
                'Total_Living_Cost': [total_living_cost],
                'Tuition_to_Rent_Ratio': [tuition_to_rent_ratio],
                'Cost_per_Year': [cost_per_year]
            })
            
            # Make predictions
            tca_pred = tca_model.predict(input_data)[0]
            affordability = affordability_model.predict(input_data)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Total Cost", f"${tca_pred:,.2f}")
            with col2:
                st.metric("Affordability Tier", affordability)
            
            # Cost breakdown visualization
            st.subheader("Estimated Cost Breakdown")
            cost_components = {
                'Tuition': pred_tuition,
                'Rent (Annual)': input_data['Rent_USD'].values[0] * 12,
                'Insurance': input_data['Insurance_USD'].values[0],
                'Visa': input_data['Visa_Fee_USD'].values[0]
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(cost_components.keys()),
                values=list(cost_components.values()),
                hole=.3
            )])
            fig.update_layout(title="Cost Components")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • [GitHub Repository](https://github.com/yourusername/SDS-CP030-edu-spend)") 