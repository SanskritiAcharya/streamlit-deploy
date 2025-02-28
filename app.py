import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Document from Sanskriti.csv")  # Ensure correct path
    
    # Convert Avg Daily Rate to numeric (remove symbols if needed)
    df["Avg Daily Rate"] = df["Avg Daily Rate"].replace('[\$,]', '', regex=True).astype(float)
    
    # Encode categorical variables
    label_encoders = {}
    for col in ["Deposit Type", "Customer Type", "Distribution Channel", "Country"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Hotel Booking Analysis")
page = st.sidebar.radio("Navigation", ["Home", "EDA", "Predict Cancellation"])

if page == "Home":
    st.title("Hotel Booking Cancellation Prediction")
    st.write("This app predicts whether a hotel booking will be canceled based on various features using Logistic Regression.")
    st.write("Use the sidebar to navigate between pages.")

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.write("Understanding the dataset.")
    
    if st.checkbox("Show Raw Data"):
        st.write(df.head())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature to visualize", numeric_df.columns)
    fig = px.histogram(df, x=feature)
    st.plotly_chart(fig)

elif page == "Predict Cancellation":
    st.title("Predict Hotel Booking Cancellation")
    st.write("Enter the details below to predict if a booking will be canceled.")
    
    # Define user input fields based on dataset features
    col1, col2 = st.columns(2)
    with col1:
        lead_time = st.number_input("Lead Time", min_value=0, max_value=500, value=100)
        nights = st.number_input("Nights", min_value=0, max_value=30, value=3)
        guests = st.number_input("Guests", min_value=1, max_value=10, value=2)
        avg_daily_rate = st.number_input("Avg Daily Rate", min_value=0.0, max_value=1000.0, value=100.0)
    
    with col2:
        deposit_type = st.selectbox("Deposit Type", df["Deposit Type"].unique())
        customer_type = st.selectbox("Customer Type", df["Customer Type"].unique())
        distribution_channel = st.selectbox("Distribution Channel", df["Distribution Channel"].unique())
        country = st.selectbox("Country", df["Country"].unique())
    
    # Model training
    df = df.dropna()
    X = df[["Lead Time", "Nights", "Guests", "Avg Daily Rate", "Deposit Type", "Customer Type", "Distribution Channel", "Country"]]
    y = df["Cancelled (0/1)"]  # Ensure correct target variable name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
    
    # Predicting new input
    user_data = np.array([[lead_time, nights, guests, avg_daily_rate, deposit_type, customer_type, distribution_channel, country]])
    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)[0]
    
    result = "Canceled" if prediction == 1 else "Not Canceled"
    st.subheader(f"Prediction: {result}")
    
    # Visualization
    fig = px.bar(x=["Not Canceled", "Canceled"], y=[1-prediction, prediction], labels={'x': "Outcome", 'y': "Probability"}, title="Prediction Probability")
    st.plotly_chart(fig)
