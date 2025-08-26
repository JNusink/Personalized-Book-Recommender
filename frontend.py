import streamlit as st
import pandas as pd
import requests


@st.cache_data
def load_user_ids():
    try:
        df = pd.read_parquet('books_5core_1M.parquet')
        if 'user_id' not in df.columns:
            st.error("Column 'user_id' not found in the dataset")
            return ["AE224GVO7OHTYF26U6ER6BEVIUAQ"]
        user_ids = df['user_id'].dropna().unique().tolist()
        st.write(f"Loaded {len(user_ids)} unique user IDs")
        return user_ids
    except Exception as e:
        st.error(f"Error loading user IDs: {e}")
        return ["AE224GVO7OHTYF26U6ER6BEVIUAQ"]


st.title("Book Recommender")
st.markdown("Select a user ID to get personalized book recommendations.")

user_ids = load_user_ids()
selected_user_id = st.selectbox("Select User ID", user_ids, index=0)

if st.button("Get Recommendations"):
    response = requests.post(f"http://localhost:8000/predict?user_id={selected_user_id}")
    if response.status_code == 200:
        data = response.json()
        st.success(f"Recommendations for {data['user_id']}: {data['recommended_books']}")
    else:
        st.error(f"Error: {response.text}")

st.markdown("---")
st.caption("Powered by Streamlit and FastAPI")
