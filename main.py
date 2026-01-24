# main.py (Streamlit App - Updated to directly call LangGraph workflow)

import streamlit as st 
import time 
import asyncio 
import os

# Import the compiled LangGraph workflow from blog_agent.py 
from blog_agent import app, GraphState # Import app and GraphState 

# --- Page Configuration ---
st.set_page_config(
    page_title="AI YouTube Blogger",
    page_icon="🤖",
    layout="wide",
)

# --- Streamlit User Interface ---
st.title("🤖 AI YouTube Blogger")
st.markdown("Turn any YouTube video into a comprehensive, well-structured blog post using a multi-agent AI system.")

st.divider()

# Important: Provide instructions for API key
st.info("""
    **Heads up!** For this app to work, you need to set your `GROQ_API_KEY` as a Streamlit Secret.
    Go to `☰ > Settings > Manage app > Secrets` and add `GROQ_API_KEY = "your_groq_api_key_here"`.
    
    **Note on performance:** This app downloads audio and uses local AI models (Whisper, HuggingFace embeddings).
    This can be resource-intensive and slow on Streamlit Cloud. For faster results, consider using
    cloud-based transcription/embedding services or smaller models.
    Also, `yt-dlp` (used for downloading YouTube audio) needs to be available in the environment.
    If deployment fails, you might need a `packages.txt` file with `yt-dlp` or `ffmpeg`.
""")

# Input box for the YouTube URL
youtube_url = st.text_input(
    "Enter the YouTube URL below:",
    placeholder="e.g., https://www.youtube.com/watch?v=...",
)

# Submit button
submit_button = st.button("Generate Blog Post", type="primary")

if submit_button and youtube_url:
    # Basic URL validation
    if "youtube.com/watch?v=" not in youtube_url and "youtu.be/" not in youtube_url:
        st.error("Please enter a valid YouTube video URL.")
    else:
        # Show a spinner while the backend is working
        with st.spinner("The AI agents are hard at work... This may take a few minutes..."):
            start_time = time.time()
            
            try:
                # Directly invoke the LangGraph app
                # LangGraph app.invoke expects a dictionary matching the input_type
                # In our case, GraphState expects 'youtube_url'
                final_state = asyncio.run(app.ainvoke({"youtube_url": youtube_url}))
                
                blog_post = final_state.get('draft_blog', 'Error: Could not generate blog post.')
                
                end_time = time.time()
                processing_time = end_time - start_time

                # Display the results
                st.success(f"Blog post generated in {processing_time:.2f} seconds!")
                st.divider()
                
                # Display the final blog post
                st.subheader("Your Generated Blog Post:")
                st.markdown(blog_post)

                # Optionally display revision history for debugging/insight
                if final_state.get('revision_history'):
                    st.subheader("Revision History:")
                    for i, rev in enumerate(final_state['revision_history']):
                        st.text(f"Revision {i+1}: {rev[:200]}...") # Show first 200 chars

            except Exception as e:
                st.error(f"An error occurred during blog generation: {e}")
                st.info("Please check the console logs for more details (if on Streamlit Cloud, click 'Manage app').")

elif submit_button:
    st.warning("Please enter a YouTube URL.")

