# main.py
import os
import streamlit as st
from ClimateRAGChatbot import load_or_initialize_rag, ClimateDocument
import sys

import functions
sys.modules['__main__'].ClimateDocument = functions.ClimateDocument

# Set your Gemini API key (or load from .env, etc.)
# GEMINI_API_KEY = "AIzaSyCF4kAqP2Y8ZB8Xjyhbp0LXMlxEidpPomk"
GEMINI_API_KEY = "AIzaSyDJGIusm7fXTxBXC0f4oW-15YKfhDSahC8"

TOP_50_CITIES = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "San Francisco, CA",
    "Charlotte, NC", "Indianapolis, IN", "Seattle, WA", "Denver, CO", "Washington, DC",
    "Boston, MA", "El Paso, TX", "Detroit, MI", "Nashville, TN", "Portland, OR",
    "Memphis, TN", "Oklahoma City, OK", "Las Vegas, NV", "Louisville, KY", "Baltimore, MD",
    "Milwaukee, WI", "Albuquerque, NM", "Tucson, AZ", "Fresno, CA", "Sacramento, CA",
    "Mesa, AZ", "Kansas City, MO", "Atlanta, GA", "Miami, FL", "Cleveland, OH",
    "New Orleans, LA", "Minneapolis, MN", "Tampa, FL", "Orlando, FL", "Pittsburgh, PA",
    "Cincinnati, OH", "St. Louis, MO", "Raleigh, NC", "Salt Lake City, UT", "Buffalo, NY"
    ]

# Load the RAG system (this may take a few seconds)
rag = load_or_initialize_rag(TOP_50_CITIES, GEMINI_API_KEY, "climate_rag_50_cities")

# Streamlit page settings
st.set_page_config(page_title=" Climate RAG Chatbot", layout="centered")
st.title("Climate Analysis Assistant")
st.write("Ask climate-related questions, like temperature trends or rainfall in major US cities.")

# User input
query = st.text_input(" Ask a question:", placeholder="e.g., What was the temperature in Phoenix in July 2023?")

# Submit button
if st.button("Get Answer") and query:
    with st.spinner("Processing..."):
        result = rag.query(query)

    st.markdown("---")
    st.markdown("### Answer")
    st.write(result["answer"])

    if result.get("locations"):
        st.markdown("### Locations Identified")
        st.write(", ".join(result["locations"]))

    if result.get("sources"):
        st.markdown("### Retrieved Sources")
        for i, src in enumerate(result["sources"]):
            with st.expander(f"Source {i+1}"):
                st.markdown(src["content"])
