# blog_agent.py (Refactored from server.py)

import os 
import glob  
import subprocess 
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

# No FastAPI imports here anymore

from langgraph.graph import StateGraph, END
import operator
from operator import itemgetter # For LCEL
from langchain_core.runnables import RunnablePassthrough # For LCEL

# --- Tool & LangChain Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from transformers import pipeline

# --- 1. SETUP: Environment Variables and LLMs ---
# Load .env file only if not running in Streamlit Cloud (where secrets are used)
if not os.getenv("STREAMLIT_CLOUD"): # Assuming Streamlit Cloud sets an env var
    load_dotenv()

# IMPORTANT: Ensure GROQ_API_KEY is set as a Streamlit Secret or in your .env file
# if running locally.
if not os.getenv("GROQ_API_KEY"):
    # In Streamlit Cloud, this should come from secrets.toml
    # For local testing, ensure it's in .env
    print("WARNING: GROQ_API_KEY environment variable not set. LLM calls may fail.")

llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

# Note: Loading local models like HuggingFaceEmbeddings and Whisper can be
# memory-intensive and slow on Streamlit Cloud. Consider cloud-based alternatives
# or smaller models if you encounter resource limits.
print("Loading local embeddings model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings model loaded.")

try:
    print("Loading local transcription model (whisper-base.en)...")
    transcription_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    print("Transcription model loaded.")
except Exception as e:
    print(f"ERROR: Could not load the local transcription model. This will prevent transcription. Error: {e}")
    # Do not exit, allow the app to run but transcription will fail
    transcription_pipe = None


# --- 2. DEFINE GRAPH STATE ---
class GraphState(TypedDict):
    youtube_url: str
    transcript: str
    titles: list[str]
    chosen_title: str
    draft_blog: str
    critique: str
    revision_history: Annotated[list[str], operator.add]


# --- 3. DEFINE GRAPH NODES (THE "AGENTS") ---

def transcribe_video(state: GraphState):
    """Node 1: Transcribes the YouTube video using a FREE, local Whisper model."""
    print("--- AGENT: Transcriber (using LOCAL Whisper model) ---")
    if not transcription_pipe:
        print("Transcription model not loaded. Skipping transcription.")
        return {"transcript": "Transcription failed: Model not loaded.", "revision_history": ["Transcription failed"]}

    url = state['youtube_url']
    output_template = "temp_audio.%(ext)s"
    
    # Clean up old files
    for old_file in glob.glob("temp_audio.*"):
        os.remove(old_file)

    command = [
        "yt-dlp", "-f", "bestaudio/best", "--no-playlist",
        "-o", output_template, url,
    ]

    try:
        print(f"Running command: {' '.join(command)}")
        # Adding timeout for subprocess to prevent hanging
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)

        audio_files = glob.glob("temp_audio.*")
        if not audio_files:
            raise FileNotFoundError("yt-dlp ran, but the audio file was not found.")
        
        audio_file_path = audio_files[0]
        print(f"Audio downloaded successfully: {audio_file_path}")

        print("Transcribing audio locally... (This might take a moment)")
        result = transcription_pipe(audio_file_path, return_timestamps=True)
        transcript_text = result["text"]
        
        os.remove(audio_file_path)
        
        print("--- Transcription Complete ---")
        return {"transcript": transcript_text, "revision_history": []}

    except subprocess.CalledProcessError as e:
        print(f"ERROR: yt-dlp failed to download the audio. Stderr: {e.stderr}")
        return {"transcript": f"Transcription failed: yt-dlp error: {e.stderr}", "revision_history": [f"Transcription failed: yt-dlp error: {e.stderr}"]}
    except FileNotFoundError as e:
        print(f"ERROR: Audio file not found after yt-dlp. Error: {e}")
        return {"transcript": f"Transcription failed: Audio file not found. Error: {e}", "revision_history": [f"Transcription failed: Audio file not found. Error: {e}"]}
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: yt-dlp command timed out. Error: {e}")
        return {"transcript": f"Transcription failed: yt-dlp timed out. Error: {e}", "revision_history": [f"Transcription failed: yt-dlp timed out. Error: {e}"]}
    except Exception as e:
        print(f"An unexpected error occurred in the transcription node: {e}")
        return {"transcript": f"Transcription failed: Unexpected error: {e}", "revision_history": [f"Transcription failed: Unexpected error: {e}"]}


def generate_titles(state: GraphState):
    """Node 2: Generates catchy titles for the blog."""
    print("--- AGENT: Titleist ---")
    transcript = state['transcript']
    
    if "Transcription failed" in transcript: # Check if transcription failed previously
        print("Skipping title generation due to failed transcription.")
        return {"titles": ["Error: Transcription Failed"], "chosen_title": "Error: Transcription Failed"}

    prompt = ChatPromptTemplate.from_template(
        "You are an expert SEO and copywriter. Based on the following video transcript, "
        "generate a JSON object with a single key 'titles' containing a list of 5 "
        "catchy, viral, and SEO-friendly blog post titles.\n\n"
        "Transcript:\n{transcript}"
    )
    
    try:
        title_gen_chain = prompt | llm | JsonOutputParser()
        titles_result = title_gen_chain.invoke({"transcript": transcript})
        
        # Ensure titles_result is a dictionary with a 'titles' key that is a list
        if isinstance(titles_result, dict) and 'titles' in titles_result and isinstance(titles_result['titles'], list):
            print(f"--- Titles Generated: {titles_result['titles']} ---")
            return {"titles": titles_result['titles'], "chosen_title": titles_result['titles'][0]}
        else:
            print(f"Unexpected title generation format: {titles_result}")
            return {"titles": ["Generated Title Error"], "chosen_title": "Generated Title Error"}
    except Exception as e:
        print(f"Error in title generation: {e}")
        return {"titles": [f"Title Generation Failed: {e}"], "chosen_title": f"Title Generation Failed: {e}"}


def write_blog_rag(state: GraphState):
    """Node 3: Writes the blog post using a RAG approach."""
    print("--- AGENT: RAG Blog Writer ---")
    transcript = state['transcript']
    title = state['chosen_title']

    if "Transcription failed" in transcript or "Title Generation Failed" in title:
        print("Skipping blog writing due to previous errors.")
        return {"draft_blog": "Blog generation skipped due to prior errors."}
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_text(transcript)
    
    print("Creating vectorstore with local embeddings...")
    vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    print("Vectorstore created.")

    rag_prompt = ChatPromptTemplate.from_template(
        "You are a professional tech blogger. Your task is to write a comprehensive blog post on the given title, "
        "using the provided context from a video transcript as your primary source. "
        "Structure the post with a clear introduction, logical sections with headings, and a conclusion. "
        "Use bullet points for key takeaways. Your tone should be engaging and expert.\n\n"
        "Title: {title}\n\n"
        "Context from transcript (use this to inform your writing):\n{context}"
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("title") | retriever | format_docs
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        print("Generating blog post draft...")
        # Pass the full state to invoke, as itemgetter("title") expects it
        draft_blog = rag_chain.invoke({"title": title, "transcript": transcript})
        print("Blog post draft generated.")
        
        return {"draft_blog": draft_blog}
    except Exception as e:
        print(f"Error in RAG blog writing: {e}")
        return {"draft_blog": f"Blog writing failed: {e}"}


def critique_blog(state: GraphState):
    """Node 4: Critiques the draft blog post (MCP step)."""
    print("--- AGENT: Editor/Critique ---")
    draft = state['draft_blog']

    if "Blog generation skipped" in draft or "Blog writing failed" in draft:
        print("Skipping critique due to failed blog generation.")
        return {"critique": "Critique skipped due to prior blog generation errors.", "revision_history": ["Critique skipped"]}
    
    prompt = ChatPromptTemplate.from_template(
        "You are a master editor. Review the blog post draft below. "
        "Your goal is to make it publication-ready. Provide constructive criticism and suggestions. "
        "If the blog is excellent and ready to publish, respond ONLY with the word 'LGTM' (Looks Good To Me). "
        "Otherwise, provide a numbered list of specific, actionable feedback points.\n\n"
        "Draft:\n{draft}"
    )
    
    try:
        critique_chain = prompt | llm | StrOutputParser()
        critique = critique_chain.invoke({"draft": draft})
        
        print(f"--- Critique Received: {critique} ---")
        return {"critique": critique, "revision_history": [f"Critique: {critique}"]}
    except Exception as e:
        print(f"Error in critiquing blog: {e}")
        return {"critique": f"Critique failed: {e}", "revision_history": [f"Critique failed: {e}"]}


def rewrite_blog(state: GraphState):
    """Node 5: Rewrites the blog based on the critique."""
    print("--- AGENT: Re-Writer ---")
    draft = state['draft_blog']
    critique = state['critique']

    if "Critique failed" in critique:
        print("Skipping rewrite due to failed critique.")
        return {"draft_blog": "Rewrite skipped due to prior critique errors.", "revision_history": ["Rewrite skipped"]}
    
    prompt = ChatPromptTemplate.from_template(
        "You are a blog writer. You have received feedback on your draft. "
        "Rewrite the blog post, carefully incorporating the following critique to improve it.\n\n"
        "Original Draft:\n{draft}\n\n"
        "Critique:\n{critique}\n\n"
        "Please provide the new, improved version of the blog post:"
    )
    
    try:
        rewrite_chain = prompt | llm | StrOutputParser()
        new_draft = rewrite_chain.invoke({"draft": draft, "critique": critique})
        
        print("--- Blog Post Rewritten ---")
        return {"draft_blog": new_draft, "revision_history": [f"Rewritten Draft:\n{new_draft}"]}
    except Exception as e:
        print(f"Error in rewriting blog: {e}")
        return {"draft_blog": f"Rewrite failed: {e}", "revision_history": [f"Rewrite failed: {e}"]}


# --- 4. DEFINE CONDITIONAL EDGES ---
def should_continue(state: GraphState):
    """Conditional Edge: Decides whether to end the process or loop for revisions."""
    print("--- DECISION: Reviewing Critique ---")
    critique = state['critique']
    
    # Check for hard failures in previous steps
    if "Transcription failed" in state.get('transcript', '') or \
       "Title Generation Failed" in state.get('chosen_title', '') or \
       "Blog writing failed" in state.get('draft_blog', '') or \
       "Critique failed" in state.get('critique', ''):
        print("--- Decision: Previous step failed. Ending process. ---")
        return "end"

    # Limit revisions to prevent infinite loops
    if len(state['revision_history']) > 4:
        print("--- Decision: Max revisions reached. Ending process. ---")
        return "end"
    
    if "LGTM" in critique:
        print("--- Decision: Blog approved. Ending process. ---")
        return "end"
    else:
        print("--- Decision: Blog needs revision. Continuing to rewriter. ---")
        return "continue"


# --- 5. ASSEMBLE THE GRAPH ---
workflow = StateGraph(GraphState)

workflow.add_node("transcriber", transcribe_video)
workflow.add_node("title_generator", generate_titles)
workflow.add_node("rag_writer", write_blog_rag)
workflow.add_node("critique", critique_blog)
workflow.add_node("rewriter", rewrite_blog)

workflow.set_entry_point("transcriber")
workflow.add_edge("transcriber", "title_generator")
workflow.add_edge("title_generator", "rag_writer")
workflow.add_edge("rag_writer", "critique")
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {"continue": "rewriter", "end": END},
)
workflow.add_edge("rewriter", "critique")

# Compile the workflow into a runnable app
app = workflow.compile()

# No FastAPI serving in this file anymore
