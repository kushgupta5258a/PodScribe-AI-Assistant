🎙️ PodScribe: AI-Powered Podcast Assistant
PodScribe is a web application built with Streamlit that allows you to transcribe, summarize, and interact with audio content using the power of AI.

Features
Accurate Transcription: Upload an audio file (.mp3, .wav, etc.) and get a full transcript powered by OpenAI's Whisper model.

Intelligent Summarization: Get a concise, bullet-pointed summary of the key topics discussed.

Detailed Insights: Extract structured information like main topics, key takeaways, and mentioned resources.

Sentiment Analysis: Understand the overall tone and sentiment of the conversation.

Speaker Identification: Automatically label different speakers to make the transcript easy to read.

Interactive Q&A: Chat with the podcast! Ask specific questions about the content and get instant answers.

Clickable Transcript: Click on any word in the transcript to jump to that exact moment in the audio player.

Tech Stack
Frontend: Streamlit

Transcription: openai-whisper

AI Analysis: Various LLMs (like openai/gpt-4o-mini) accessed via the OpenRouter API.

Setup and Installation
1. Clone the repository:

git clone <your-repo-url>
cd <your-repo-folder>

2. Create and activate a virtual environment:

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
Make sure you have ffmpeg installed on your system. Then, install the required Python libraries from the requirements.txt file:

pip install -r requirements.txt

4. Set up your API key:

Create a free account on OpenRouter.ai and get an API key.

Create a folder in your project named .streamlit.

Inside that folder, create a file named secrets.toml.

Add your API key to the secrets.toml file:

OPENROUTER_API_KEY = "sk-or-..."

5. Run the application:

streamlit run app.py

The application will open in your web browser. Enjoy!