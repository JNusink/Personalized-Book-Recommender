import streamlit as st
import boto3
import matplotlib.pyplot as plt

st.title("Model Monitoring Dashboard")
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('BookRecommendations')
items = table.scan()['Items']
st.write("Recent Prediction Logs:", items)
if items:
    latencies = [float(item['latency']) for item in items if 'latency' in item]
    if latencies:
        st.write(f"Average Latency: {sum(latencies) / len(latencies):.4f} seconds")
        fig, ax = plt.subplots()
        ax.hist(latencies, bins=10, edgecolor='black')
        ax.set_title("Latency Distribution")
        ax.set_xlabel("Latency (seconds)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)