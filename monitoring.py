import streamlit as st
import boto3

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('BookRecommendations')

st.title("Model Monitoring Dashboard")

# Fetch recent predictions
response = table.scan(
    Limit=100  # Limit to 100 items for performance
)
items = response['Items']

# Calculate average latency
latencies = [float(item['latency']) for item in items if 'latency' in item]
avg_latency = sum(latencies) / len(latencies) if latencies else 0

# Rating distribution
ratings = [float(item['predicted_rating']) for item in items if 'predicted_rating' in item]
rating_counts = {i: ratings.count(i) for i in set(ratings)}

# Display metrics
st.header("Monitoring Metrics")
st.metric("Average Prediction Latency", f"{avg_latency:.2f} seconds")
st.subheader("Predicted Rating Distribution")
st.bar_chart(rating_counts)
