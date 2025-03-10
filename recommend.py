import streamlit as st
import joblib
import pandas as pd

# Load models
predicted_ratings_df = joblib.load("predicted_ratings.pkl")
knn_model = joblib.load("knn_model.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
user_item_matrix = joblib.load("user_item_matrix.pkl")
df = pd.read_csv('tourism_data.csv')

# Collaborative Filtering Recommendation
def collaborative_recommend(user_id, preferred_city=None, preferred_type=None, n=5):
    if user_id not in predicted_ratings_df.index:
        return []
    sorted_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False)


    recommendations = []
    seen_attractions = set()

    # Get attractions user already rated
    user_rated_attractions = set(df[df['UserId'] == user_id]['Attraction'])

    for aid in sorted_ratings.index:
        attraction_rows = df[df['AttractionId'] == aid]
        if attraction_rows.empty:
            continue

        for _, attraction_row in attraction_rows.iterrows():
            formatted_attraction = f"{attraction_row['Attraction']} ({attraction_row['CityName']})"

            # Skip if user already rated it
            if attraction_row['Attraction'] in user_rated_attractions:
                continue  
            
            # Apply filters
            if preferred_city and attraction_row['CityName'] != preferred_city:
                continue
            if preferred_type and attraction_row['AttractionType'] != preferred_type:
                continue

            recommendations.append(formatted_attraction)
            seen_attractions.add(attraction_row['Attraction'])  

            if len(recommendations) >= n:
                return recommendations

    # If CF fails, return popular attractions as backup
    if len(recommendations) < n:
        popular_attractions = (
            df[df['CityName'] == preferred_city]
            .groupby("Attraction")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        
        for att in popular_attractions:
            if att not in seen_attractions:
                recommendations.append(f"{att} ({preferred_city})")
            if len(recommendations) >= n:
                break

    return recommendations


# Content-Based Recommendation
def content_recommend(attraction_name, preferred_city=None, preferred_type=None, n=5):
    attraction_name = attraction_name.lower().strip()
    if attraction_name not in df['Attraction'].str.lower().str.strip().values:
        return []
    idx = df[df['Attraction'].str.lower().str.strip() == attraction_name].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=20)
    recommended_attractions = []
    for i in indices.flatten()[1:]:  # Skip the first one (itself)
        attraction_row = df.iloc[i]
        if preferred_city and attraction_row['CityName'] != preferred_city:
            continue
        if preferred_type and attraction_row['AttractionType'] != preferred_type:
            continue
        formatted_attraction = f"{attraction_row['Attraction']} ({attraction_row['CityName']})"
        if formatted_attraction not in recommended_attractions:
            recommended_attractions.append(formatted_attraction)
        if len(recommended_attractions) >= n:
            break
    
    # Fallback to return at least n recommendations
    if len(recommended_attractions) < n:
        for i in indices.flatten()[1:]:
            attraction_row = df.iloc[i]
            formatted_attraction = f"{attraction_row['Attraction']} ({attraction_row['CityName']})"
            if formatted_attraction not in recommended_attractions:
                recommended_attractions.append(formatted_attraction)
            if len(recommended_attractions) >= n:
                break
    
    return recommended_attractions



# Hybrid Recommendation
def hybrid_recommend(user_id, attraction_name, preferred_city=None, preferred_type=None, cf_weight=0.5, cb_weight=0.5, n=5):
    cf_recommendations = collaborative_recommend(user_id, preferred_city, preferred_type, n=n)
    cb_recommendations = content_recommend(attraction_name, preferred_city, preferred_type, n=n)

    # Combine both, ensuring variety
    hybrid_results = list(set(cf_recommendations[:int(cf_weight * n)] + cb_recommendations[:int(cb_weight * n)]))

    if len(hybrid_results) < n:
        extra_items = list(set(cf_recommendations + cb_recommendations) - set(hybrid_results))
        hybrid_results.extend(extra_items[: (n - len(hybrid_results))])

    return hybrid_results[:n]

# 🎨 Streamlit UI
st.title("🏝️ Tourism Experience Recommendation System")

st.sidebar.header("🔍 Search Preferences")
user_id = st.sidebar.text_input("Enter User ID:")

if user_id.isdigit():
    user_id = int(user_id)
else:
    user_id = None

attraction_name = st.sidebar.selectbox("Select Attraction:", df["Attraction"].unique())
preferred_city = st.sidebar.selectbox("Select Preferred City:", ["Any"] + list(df["CityName"].unique()))
preferred_type = st.sidebar.selectbox("Select Attraction Type:", ["Any"] + list(df["AttractionType"].unique()))

recommendation_type = st.sidebar.radio("Recommendation Type", ["Collaborative Filtering", "Content-Based", "Hybrid"])

if st.sidebar.button("Get Recommendations"):
    st.subheader("🔮 Recommended Attractions:")

    if recommendation_type == "Collaborative Filtering":
        recommendations = collaborative_recommend(user_id, preferred_city if preferred_city != "Any" else None, preferred_type if preferred_type != "Any" else None)
    elif recommendation_type == "Content-Based":
        recommendations = content_recommend(attraction_name, preferred_city if preferred_city != "Any" else None, preferred_type if preferred_type != "Any" else None)
    else:
        recommendations = hybrid_recommend(user_id, attraction_name, preferred_city if preferred_city != "Any" else None, preferred_type if preferred_type != "Any" else None)

    for rec in recommendations:
        st.write(f"- {rec}")
