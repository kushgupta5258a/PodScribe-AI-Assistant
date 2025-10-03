import streamlit as st
import whisper
import os
from openai import OpenAI
from openai import APIError # --- NEW: Import specific APIError for better handling ---

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="PodScribe",
    page_icon="🎙️"
)

# --- Custom CSS for a new, high-contrast dark theme ---
st.markdown("""
<style>
    /* Main background color */
    .stApp {
        background-color: #0E1117; 
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Card-like containers */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 25px;
        background-color: #161B22;
    }
    
    /* Main button styling - a vibrant blue */
    .stButton > button {
        background-color: #0078F2;
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: background-color 0.2s, transform 0.2s;
    }
    .stButton > button:hover {
        background-color: #0056B3;
        transform: scale(1.02);
    }
    
    /* Header and text styling */
    h1, h2, h3, .stMarkdown, .stFileUploader, .stSelectbox {
        color: #C9D1D9; /* A light grey for readability */
    }
    h1, h2, h3 {
        color: #FFFFFF; /* Pure white for main headers */
    }

    /* Custom styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-bottom: 2px solid #30363D;
        color: #8B949E; /* Muted color for inactive tabs */
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #0078F2;
        color: #FFFFFF;
    }

    /* Interactive transcript hover color */
    .interactive-word:hover {
        background-color: #0078F2 !important;
        color: white !important;
        border-radius: 3px;
    }
    
    /* Make info boxes stand out */
    [data-testid="stInfo"] {
        background-color: rgba(0, 120, 242, 0.1);
        border: 1px solid #0078F2;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)


# --- App Configuration & API Keys ---
SUPPORTED_LANGUAGES = {
    "Auto-Detect": None, "English": "en", "Hindi": "hi", "Spanish": "es",
    "French": "fr", "German": "de", "Japanese": "ja", "Korean": "ko", "Urdu": "ur",
}
TEMP_AUDIO_FILENAME = "temp_audio_file.mp3"

try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )
except Exception as e:
    st.error("Error: OpenRouter API key not found. Please add it to your .streamlit/secrets.toml file.", icon="🚨")
    st.stop()

# --- AI & Transcription Functions (Cached) ---
@st.cache_data
def transcribe_audio(file_path, language_code, model_name):
    st.info(f"Performing first-time transcription with the '{model_name}' model...")
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, language=language_code, fp16=False, word_timestamps=True)
    return result

@st.cache_data
def get_summary(transcript):
    """Generates a summary using a model on OpenRouter."""
    st.info("Generating summary via OpenRouter...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert podcast summarizer. Provide a concise, easy-to-read summary of the following transcript. Use bullet points to highlight the key topics and discussions."},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
    )
    return response.choices[0].message.content

@st.cache_data
def get_insights(transcript):
    """Generates structured insights from the transcript."""
    st.info("Extracting key insights via OpenRouter...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert analyst. Analyze the following transcript and provide a structured breakdown. Your output must be in Markdown format.

Provide the following sections:
- **Main Topics:** A bulleted list of the 5 main topics discussed, with a brief one-sentence description for each.
- **Key Takeaways:** A bulleted list of the most important takeaways or action items mentioned.
- **Mentioned Resources:** A bulleted list of any people, books, or products mentioned.

Format each section with a clear, bolded heading (e.g., '**Main Topics**')."""
            },
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
    )
    return response.choices[0].message.content


@st.cache_data
def get_sentiment(transcript):
    """Analyzes the sentiment of the transcript."""
    st.info("Analyzing sentiment via OpenRouter...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a sentiment analysis expert. Analyze the overall sentiment of the following transcript. 

Your response must be a single sentence in the following format:
**Overall Sentiment:** [Positive/Negative/Neutral/Mixed], with [a brief justification].

Example:
**Overall Sentiment:** Positive, with the speakers expressing optimism about future technology trends."""
            },
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
    )
    return response.choices[0].message.content

@st.cache_data
def get_suggested_questions(transcript):
    """Generates a few insightful questions based on the transcript."""
    st.info("Generating suggested questions via OpenRouter...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Based on the provided transcript, generate 3 interesting and insightful questions that a user might want to ask about the content. Your output must be a Markdown bulleted list."},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
    )
    return response.choices[0].message.content

@st.cache_data
def get_diarized_transcript(transcript):
    """Identifies and labels speakers in the transcript using an LLM."""
    st.info("Identifying speakers via OpenRouter...")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert transcript editor. Your task is to analyze the following transcript and reformat it by identifying and labeling the different speakers.

Use labels like 'Speaker A:', 'Speaker B:', etc.
If there is only one speaker, do not add any labels.
Ensure each speaker's turn starts on a new line."""
            },
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
    )
    return response.choices[0].message.content

def get_qa_response(transcript, question):
    """Answers a user's question based on the transcript."""
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful Q&A assistant for a podcast. Your task is to answer the user's questions based ONLY on the provided transcript. 
                Do not use any external knowledge. If the answer is not found in the transcript, you must say 'I'm sorry, but that information is not available in the podcast transcript.'"""
            },
            {"role": "user", "content": f"Here is the podcast transcript:\n\n{transcript}\n\nNow, please answer my question:\n{question}"},
        ],
    )
    return response.choices[0].message.content

# --- Helper Functions ---
def format_chat_history(messages):
    """Formats the chat history list into a readable string."""
    history_str = "Your PodScribe Chat History\n"
    history_str += "="*30 + "\n\n"
    for msg in messages:
        history_str += f"**{msg['role'].capitalize()}:** {msg['content']}\n\n"
    return history_str

def create_interactive_transcript(transcription_result):
    """Generates an HTML string for a clickable transcript."""
    html = '<div style="line-height: 2.0; font-size: 16px; color: #C9D1D9;">' # Set default text color
    for segment in transcription_result.get('segments', []):
        for word in segment.get('words', []):
            start_time = word['start']
            word_text = word['word']
            
            html += (
                f'<span class="interactive-word" style="cursor: pointer; padding: 2px;" '
                f'onclick="document.querySelector(\'audio\').currentTime={start_time}; document.querySelector(\'audio\').play();">'
                f'{word_text} </span>'
            )
    html += '</div>'
    return html


# --- Main App ---
with st.sidebar:
    st.header("Upload & Configure")
    uploaded_file = st.file_uploader(
        "Upload an audio file", type=["mp3", "mp4", "wav", "m4a"]
    )
    if uploaded_file is not None:
        # --- NEW: Check if a new file has been uploaded ---
        # This ensures we reset the analysis if the user changes the file.
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.analysis_complete = False
            st.session_state.last_uploaded_file = uploaded_file.name

        st.audio(uploaded_file)
        selected_language = st.selectbox(
            "Audio Language:",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=1
        )
        if st.button("Start Analysis ✨"):
            # Initialize session state variables for a new analysis
            st.session_state.analysis_complete = False
            st.session_state.transcript_result = {}
            st.session_state.summary = ""
            st.session_state.insights = ""
            st.session_state.sentiment = ""
            st.session_state.questions = ""
            st.session_state.diarized_transcript = ""
            st.session_state.messages = []

            with st.spinner("Starting analysis... Please wait."):
                # --- MODIFIED: More specific error handling ---
                try:
                    with open(TEMP_AUDIO_FILENAME, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    model_to_use = "small" if selected_language == "Hindi" else "base"
                    
                    transcription_result = transcribe_audio(
                        TEMP_AUDIO_FILENAME,
                        SUPPORTED_LANGUAGES[selected_language],
                        model_to_use
                    )
                    
                    if transcription_result and "text" in transcription_result:
                        plain_transcript = transcription_result["text"]
                        
                        # Store all results in session state
                        st.session_state.transcript_result = transcription_result
                        st.session_state.summary = get_summary(plain_transcript)
                        st.session_state.insights = get_insights(plain_transcript)
                        st.session_state.sentiment = get_sentiment(plain_transcript)
                        st.session_state.questions = get_suggested_questions(plain_transcript)
                        st.session_state.diarized_transcript = get_diarized_transcript(plain_transcript)
                        st.session_state.analysis_complete = True
                    else:
                        st.warning("Transcription returned no text.")
                
                except APIError as e:
                    st.error(f"An API error occurred with OpenRouter: {e}. Please check your API key and credits.", icon="📡")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}", icon="🔥")
                finally:
                    if os.path.exists(TEMP_AUDIO_FILENAME):
                        os.remove(TEMP_AUDIO_FILENAME)

    st.markdown("---")
    with st.expander("About PodScribe"):
        st.write("""
            **PodScribe** is an AI-powered assistant designed to help you understand audio content faster.
            
            Simply upload a podcast, lecture, or meeting recording, and PodScribe will provide:
            - A full, accurate transcript.
            - A concise summary of the key points.
            - Detailed insights and topics.
            - An interactive chat to ask questions about the content.
            
            This tool is built with Streamlit, OpenAI's Whisper, and various LLMs accessed via OpenRouter.
        """)


st.title("🎙️ PodScribe")
st.write("Welcome! Upload an audio file in the sidebar, and let AI provide a complete analysis.")

# --- Main Content Area: Conditional Display ---
if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
    st.header("Analysis Result")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "💡 Key Insights", "📊 Sentiment Analysis", "💬 Chat"])

    with tab1:
        st.subheader("Podcast Summary")
        st.markdown(st.session_state.summary)
        st.download_button("⬇️ Download Summary", st.session_state.summary, "summary.txt")

    with tab2:
        st.subheader("Detailed Insights")
        st.markdown(st.session_state.insights)
        st.download_button("⬇️ Download Insights", st.session_state.insights, "insights.txt")

    with tab3:
        st.subheader("Sentiment Analysis")
        st.markdown(st.session_state.sentiment)

    with tab4:
        st.subheader("Ask a Question")
        with st.container(border=True):
            st.markdown("**Here are some questions you could ask:**")
            st.markdown(st.session_state.questions)
        st.markdown("---") 
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("What would you like to ask?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_qa_response(st.session_state.transcript_result["text"], prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        if len(st.session_state.messages) > 0:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun() 
            with col2:
                chat_history_str = format_chat_history(st.session_state.messages)
                st.download_button("⬇️ Download Chat History", chat_history_str, "chat_history.txt")

    st.header("Full Transcript")
    
    # Use columns to place the two transcript versions side-by-side
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Show Interactive Transcript", expanded=True):
            interactive_html = create_interactive_transcript(st.session_state.transcript_result)
            st.markdown(interactive_html, unsafe_allow_html=True)
    with col2:
        with st.expander("Show Speaker-Labeled Transcript", expanded=True):
            st.markdown(st.session_state.diarized_transcript)
else:
    st.info("Please upload a file and click 'Start Analysis' to see the results.")
