import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing objects
model = joblib.load("model.pkl")  
scaler = joblib.load("scaler.pkl")  
label_encoders = joblib.load("label_encoders.pkl")  

# Load original dataset to maintain feature order
df = pd.read_csv("tourism_data.csv")

# Get feature names used during training
expected_features = scaler.feature_names_in_.tolist()

# Extract unique values for categorical features
categorical_cols = ["Contenent", "Region", "Country", "CityName", "Attraction", "AttractionType"]
category_options = {col: df[col].unique().tolist() for col in categorical_cols}

# Streamlit UI
st.title("Tourism Visit Mode Prediction")

# Numeric Inputs
visit_year = st.slider("Visit Year", int(df["VisitYear"].min()), int(df["VisitYear"].max()), 2025)
visit_month = st.slider("Visit Month", 1, 12, 6)
rating = st.slider("User Rating (1-5)", 1.0, 5.0, 3.0)
overall_avg_rating = st.slider("Overall Avg Rating", df["Overall_Avg_Rating"].min(), df["Overall_Avg_Rating"].max(), 4.5)

# Categorical Inputs
continent = st.selectbox("Continent", category_options["Contenent"])
region = st.selectbox("Region", category_options["Region"])
country = st.selectbox("Country", category_options["Country"])
city = st.selectbox("City", category_options["CityName"])
attraction = st.selectbox("Attraction", category_options["Attraction"])
attraction_type = st.selectbox("Attraction Type", category_options["AttractionType"])

# Prepare input data as a DataFrame
input_data = pd.DataFrame([[visit_year, visit_month, rating, overall_avg_rating,
                            continent, region, country, city, attraction, attraction_type]],
                          columns=["VisitYear", "VisitMonth", "Rating", "Overall_Avg_Rating", 
                                   "Contenent", "Region", "Country", "CityName", "Attraction", "AttractionType"])

# Apply Label Encoding for categorical features
for col in categorical_cols:
    if col in label_encoders:
        if input_data[col][0] in label_encoders[col].classes_:
            input_data[col] = label_encoders[col].transform(input_data[col])
        else:
            st.warning(f"Unseen value in {col}, assigning default encoding (-1)")
            input_data[col] = -1  

# Ensure feature order matches training
input_data = input_data[expected_features]  

# Apply MinMaxScaler
input_data_scaled = scaler.transform(input_data) 

# Predict Button
if st.button("Predict Visit Mode"):
    prediction = model.predict(input_data_scaled)[0]
    visit_mode_mapping = {0: "Business", 1: "Family", 2: "Couples", 3: "Friends", 4: "Solo"}
    predicted_mode = visit_mode_mapping.get(prediction, "Unknown")

    st.success(f"Predicted Visit Mode: {predicted_mode}")
