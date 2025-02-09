import streamlit as st
import os
import time
import io
import uuid
import threading
from dotenv import load_dotenv
import pyttsx3  # For text-to-speech
import speech_recognition as sr  # For speech-to-text

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter

# For DOCX processing (install via: pip install python-docx)
try:
    import docx
except ImportError:
    docx = None

# For PDF processing, we use PyPDF2
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# ========= Initialize the TTS engine and select a professional female voice =========
tts_engine = pyttsx3.init()

# Try to choose a female voice based on available voices.
# voices = tts_engine.getProperty('voices')
# female_voice_id = None
# for voice in voices:
#     # Adjust selection logic based on your system. For Windows, "Zira" is usually a female voice.
#     if "Zira" in voice.name or "female" in voice.name.lower():
#         female_voice_id = voice.id
#         break
#
# if female_voice_id:
#     tts_engine.setProperty('voice', female_voice_id)
# else:
#     st.warning("Female voice not found, using default voice.")
#
# # Optionally, adjust TTS properties (e.g., rate, volume)
# tts_engine.setProperty('rate', 150)
# tts_engine.setProperty('volume', 0.8)


def speak_text(text):
    """Speak text using the TTS engine in a separate thread to avoid run loop errors."""
    tts_engine.say(text)
    thread = threading.Thread(target=tts_engine.runAndWait)
    thread.start()


# ========= Helper Functions =========

def read_docx(file_obj):
    """Extract text from a DOCX file."""
    try:
        document = docx.Document(file_obj)
        full_text = "\n".join([para.text for para in document.paragraphs])
        return full_text
    except Exception as e:
        return f"Error reading DOCX: {e}"


def read_pdf(file_obj):
    """Extract text from a PDF file using PyPDF2."""
    if PyPDF2 is None:
        return "PyPDF2 is not installed. Please install PyPDF2 to parse PDFs."
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


def build_temp_vectorstore(document_text, embeddings, chunk_size=1000, chunk_overlap=100):
    """Split document text into chunks and build a temporary FAISS vector store."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(document_text)
    try:
        temp_db = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"Error building temporary vector store: {e}")
        temp_db = None
    return temp_db


def transcribe_audio(audio_file):
    """Transcribe an uploaded audio file using SpeechRecognition."""
    recognizer = sr.Recognizer()
    try:
        audio_bytes = audio_file.read()
        audio_data = sr.AudioFile(io.BytesIO(audio_bytes))
        with audio_data as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        st.error(f"Could not transcribe audio: {e}")
        return None


# ========= INITIAL SETUP =========

if not os.path.exists("LEGAL-DATA"):
    os.makedirs("LEGAL-DATA")

st.set_page_config(page_title="JusticeAI", layout="wide")

# ========= CUSTOM CSS =========
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
        color: #333333;
    }
    .chat-message {
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 8px;
        max-width: 80%;
    }
    .chat-message.user {
        background-color: #d1e7dd;
        align-self: flex-end;
    }
    .chat-message.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        align-self: flex-start;
    }
    .upload-btn, .action-btn {
        background-color: #ffd0d0;
        border: none;
        padding: 0.5em 1em;
        font-size: 0.9rem;
        border-radius: 4px;
        margin: 5px 0;
    }
    .upload-btn:hover, .action-btn:hover {
        background-color: #ff6262;
        cursor: pointer;
    }
    .stChatInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #cccccc;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========= MODE SELECTION =========
mode = st.radio("Select Mode", options=["Mode 1: Regular Chatbot", "Mode 2: Chatbot With Document"], index=0)

# ========= LOAD ENVIRONMENT VARIABLES =========
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# ========= SIDEBAR: Voice Option and Chat Sessions =========
# with st.sidebar:
#     voice_enabled = st.checkbox("Enable Voice Output", value=True)

with st.sidebar.expander("🗂️ Chat Sessions", expanded=False):
    # CHAT SESSION MANAGEMENT
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
        st.session_state.current_chat_id = None


    def new_chat():
        """Creates a new chat session."""
        chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[chat_id] = {
            "name": "New Chat",
            "messages": [],
            "memory": ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True),
        }
        st.session_state.current_chat_id = chat_id


    def delete_chat(chat_id):
        """Deletes a chat session."""
        if chat_id in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[chat_id]
            if st.session_state.chat_sessions:
                st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
            else:
                st.session_state.current_chat_id = None


    if not st.session_state.chat_sessions:
        new_chat()

    session_ids = list(st.session_state.chat_sessions.keys())
    session_names = {sid: st.session_state.chat_sessions[sid]["name"] for sid in session_ids}
    if session_ids:
        selected_session = st.selectbox(
            "Select a session",
            options=session_ids,
            format_func=lambda sid: session_names[sid],
            index=session_ids.index(
                st.session_state.current_chat_id) if st.session_state.current_chat_id in session_ids else 0
        )
        st.session_state.current_chat_id = selected_session
    col_new, col_del = st.columns(2)
    with col_new:
        if st.button("New Chat", key="new_chat_btn", help="Create a new chat session"):
            new_chat()
    with col_del:
        if st.button("Delete Chat", key="delete_chat_btn", help="Delete the current session"):
            delete_chat(st.session_state.current_chat_id)

# ========= SIDEBAR FOR MODE 1 (Regular Chatbot) =========
if mode == "Mode 1: Regular Chatbot":
    st.title("JusticeAI")
    st.divider()
    with st.sidebar:
        st.markdown("---")
        st.subheader("Train the Bot")
        train_file = st.file_uploader("Upload a file to train the bot", type=["pdf", "txt", "doc", "docx"],
                                      key="train_uploader")
        if train_file is not None:
            extension = os.path.splitext(train_file.name)[1].lower()
            if extension in [".docx", ".doc"]:
                if docx is None:
                    st.error("python-docx is not installed. Please install it.")
                    content = ""
                else:
                    content = read_docx(io.BytesIO(train_file.read()))
            elif extension == ".txt":
                content = train_file.read().decode("utf-8", errors="ignore")
            elif extension == ".pdf":
                if PyPDF2 is not None:
                    content = read_pdf(train_file)
                else:
                    st.error("PyPDF2 is not installed. Please install PyPDF2 to parse PDFs.")
                    content = ""
            else:
                content = train_file.read().decode("utf-8", errors="ignore")
            file_path = os.path.join("LEGAL-DATA", train_file.name)
            with open(file_path, "wb") as f:
                f.write(train_file.getbuffer())
            st.success(f"File '{train_file.name}' saved for training.")
        st.markdown("---")
        st.subheader("💡 Query Suggestions")
        with st.container(border=True, height=200):

            st.markdown("""
            How many days of annual leave am I entitled to?\n
            Am I allowed to take maternity leave in place of my wife?\n
            What taxes do I pay if I am self-employed?\n
            Can I request the deletion of my data from a website if I did not approve it?\n
            What is the deadline for requesting a replacement for a product I am not satisfied with?\n
            Who owns the gifts that my husband and I received at our wedding?
            """)
        st.subheader("⚠️ Warning")
        with st.container(border=True):
            st.markdown("""
            _Please note that JusticeAI may make **mistakes**. For critical legal information, always **verify** with a qualified legal professional._
            """)
        st.subheader("✍️ Authors")
        with st.container(border=True):
                st.markdown("""
                [IKEHI MATTHIAS](https://www.linkedin.com/in/matthias-ikehi-3249b8261/)  
                **BABALOLA AYODEJI**
                """)

# ========= MAIN CHAT AREA =========


# For Mode 2, store the attached document and temporary vector store.
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = None
if "temp_db" not in st.session_state:
    st.session_state.temp_db = None

# ========= SET UP EMBEDDINGS =========
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ========= SET UP THE QA CHAIN =========
prompt_template = """
<s>[INST]You are a legal assistant chatbot named JusticeAI with expertise in Nigerian law.
Your task is to answer the user's question accurately and concisely.

**Instructions**:
- In Mode 1, use your internal legal knowledge and the prebuilt vector database.
- In Mode 2, answer questions solely based on the attached document.
- If asked about your origins, state that JusticeAI was designed by Ikehi Matthias and Babalola Ayodeji.

CONTEXT: {context}

QUESTION: {question}

ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

if mode == "Mode 1: Regular Chatbot":
    opening_message = """
    Hello! I am a legal assistant, and my task is to help you understand procedures and answer questions related to the following regulations:
    - [Labor Law](https://lawsofnigeria.placng.org/laws/L1.pdf)
    - [Personal Income Tax Law](https://old.firs.gov.ng/wp-content/uploads/2021/07/Personal-Income-Tax-Act.pdf)
    - [Personal Data Protection Law](https://nitda.gov.ng/wp-content/uploads/2020/11/NigeriaDataProtectionRegulation11.pdf)
    - [Consumer Protection Law](https://placng.org/i/wp-content/uploads/2019/12/Federal-Competition-and-Consumer-Protection-Act-2018.pdf)
    - [Family Law](https://nou.edu.ng/coursewarecontent/LAW%20344%20FAMILY%20LAW%20II.pdf?utm_source=chatgpt.com)

    My role is to facilitate your understanding of legal procedures and provide you with useful and accurate information.

    How can I assist you?
    """

    st.markdown(opening_message)
    try:
        db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        db = None
    if db is not None:
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192"),
        memory=st.session_state.chat_sessions[st.session_state.current_chat_id]["memory"],
        retriever=db_retriever,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
else:
    st.info("Mode 2 selected: All responses will be based solely on the attached document.")
    uploaded_file = st.file_uploader("Upload a document for query", type=["pdf", "txt", "doc", "docx"],
                                     key="mode2_uploader")
    if uploaded_file is not None:
        extension = os.path.splitext(uploaded_file.name)[1].lower()
        if extension in [".docx", ".doc"]:
            if docx is None:
                st.error("python-docx is not installed. Please install it.")
                doc_text = ""
            else:
                doc_text = read_docx(io.BytesIO(uploaded_file.read()))
        elif extension == ".txt":
            doc_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif extension == ".pdf":
            if PyPDF2 is not None:
                doc_text = read_pdf(uploaded_file)
            else:
                st.error("PyPDF2 is not installed. Please install PyPDF2 to parse PDFs.")
                doc_text = ""
        else:
            doc_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.doc_uploaded = {"name": uploaded_file.name, "content": doc_text}
        st.success(f"Document '{uploaded_file.name}' uploaded for Mode 2.")
        temp_db = build_temp_vectorstore(doc_text, embeddings)
        if temp_db is not None:
            st.session_state.temp_db = temp_db
            temp_retriever = temp_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192"),
                memory=st.session_state.chat_sessions[st.session_state.current_chat_id]["memory"],
                retriever=temp_retriever,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
        else:
            st.error("Temporary vector store could not be built. Check the document content.")
    else:
        qa_chain = None

# --- Ensure we have a valid current chat session ---
if (st.session_state.current_chat_id is None) or (
        st.session_state.current_chat_id not in st.session_state.chat_sessions):
    new_chat()

current_session = st.session_state.chat_sessions[st.session_state.current_chat_id]

# ========= DISPLAY CHAT MESSAGES FOR CURRENT SESSION =========
for msg in current_session["messages"]:
    role = msg.get("role")
    content = msg.get("content")
    with st.chat_message(role):
        st.write(content)

# ========= CHAT INPUT =========
# Option to upload a voice note.
# voice_note = st.file_uploader("Or upload a voice note", type=["wav", "mp3", "ogg"], key="voice_note")
# if voice_note is not None:
#     transcribed_text = transcribe_audio(voice_note)
#     if transcribed_text:
#         st.write("Transcribed voice note:", transcribed_text)
#         user_input = transcribed_text
#     else:
#         user_input = None
else:
    user_input = st.chat_input("Ask something about law" if mode != "Mode 2: Chatbot With Document"
                               else f"Ask something about law (Active Document: {st.session_state.doc_uploaded['name'] if st.session_state.doc_uploaded else 'No document attached.'})")

if mode == "Mode 2: Chatbot With Document" and st.session_state.doc_uploaded is None:
    st.warning("Please upload a document above before asking questions.")
    user_input = None

if user_input:
    # Display and store user's message.
    with st.chat_message("user"):
        st.write(user_input)
    current_session["messages"].append({"role": "user", "content": user_input})

    if current_session["name"] == "New Chat":
        words = user_input.split()
        new_name = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
        current_session["name"] = new_name

    if mode == "Mode 1: Regular Chatbot":
        context_used = "General legal knowledge."
    else:
        context_used = st.session_state.doc_uploaded["content"]

    if qa_chain is None:
        st.error("QA chain is not set up. Please ensure a document is uploaded for Mode 2.")
    else:
        with st.chat_message("assistant"):
            with st.status("Thinking 💡...", expanded=True):
                try:
                    result = qa_chain.invoke(input=user_input, context=context_used)
                except Exception as e:
                    result = {"answer": f"Error processing query: {e}"}
                message_placeholder = st.empty()
                full_response = ""
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")
        current_session["messages"].append({"role": "assistant", "content": result["answer"]})

        # If voice output is enabled, speak the bot's answer.
        # if voice_enabled:
        #     speak_text(result["answer"])
