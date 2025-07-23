# streamlit_app.py

import streamlit as st
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AI YouTube Blogger",
    page_icon="🤖",
    layout="wide",
)

# --- Backend API Configuration ---
# This is the URL where your LangServe backend is running.
API_URL = "http://localhost:8000/youtube-blogger/invoke"


# --- Helper Function to Call the Backend ---
def generate_blog_post(url: str):
    """
    Sends a request to the backend API to start the blog generation process.
    """
    # The payload must match the input schema defined in the backend
    payload = {
        "input": {
            "youtube_url": url
        }
    }
    print("Payload being sent:", payload)  # Debugging line to check the payload
    try:
        # Make the POST request to the backend
        response = requests.post(API_URL, json=payload, timeout=600) # Long timeout
        
        # Check for a successful response
        if response.status_code == 200:
            # The final output from LangServe is nested under the 'output' key
            # and our graph state is the value.
            result = response.json()
            # We access the final blog from the 'draft_blog' key in our graph state
            return result.get('output', {}).get('draft_blog', 'Error: Could not find blog post in response.')
        else:
            # Handle backend errors
            return f"Error from backend: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        # Handle network or connection errors
        return f"Error connecting to backend: {e}"


# --- Streamlit User Interface ---
st.title("🤖 AI YouTube Blogger")
st.markdown("Turn any YouTube video into a comprehensive, well-structured blog post using a multi-agent AI system.")

st.divider()

# Input box for the YouTube URL
youtube_url = st.text_input(
    "Enter the YouTube URL below:",
    placeholder="e.g., https://www.youtube.com/watch?v=...",
)

# Submit button
submit_button = st.button("Generate Blog Post", type="primary")

if submit_button and youtube_url:
    # Show a spinner while the backend is working
    with st.spinner("The AI agents are hard at work... This may take a few minutes..."):
        start_time = time.time()
        
        # Call the backend function
        blog_post = generate_blog_post(youtube_url)
        
        end_time = time.time()
        processing_time = end_time - start_time

    # Display the results
    st.success(f"Blog post generated in {processing_time:.2f} seconds!")
    st.divider()
    
    # Display the final blog post
    st.subheader("Your Generated Blog Post:")
    st.markdown(blog_post)

elif submit_button:
    st.warning("Please enter a YouTube URL.")
