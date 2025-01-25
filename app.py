import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv


st.set_page_config(page_title="LawGPT")
col1, col2, col3 = st.columns([1, 4, 1])
st.title("ClarityBot")
st.divider()


# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

WARNING_MESSAGE = """
_Please note that Verdict may make **mistakes**. For critical legal information, always **verify** with a qualified legal professional. ClarityBot is here to assist, not replace professional legal advice._
"""

QUERY_SUGGESTIONS = """
How many days of annual leave am I entitled to?\n
Am I allowed to take maternity leave in place of my wife?\n
What taxes do I pay if I am self-employed?\n
Can I request the deletion of my data from a website if I did not approve it?\n
What is the deadline for requesting a replacement for a product I am not satisfied with?\n
Who owns the gifts that my husband and I received at our wedding?
"""


AUTHORS = """
[IKEHI MATTHIAS](https://www.linkedin.com/in/matthias-ikehi-3249b8261/)\n
"""
# Streamlit UI setup


with st.sidebar:
    st.subheader("💡 Query Suggestions")
    with st.container(border=True, height=200):
        st.markdown(QUERY_SUGGESTIONS)

    st.subheader("⚠️ Warning")
    with st.container(border=True):
        st.markdown(WARNING_MESSAGE)

    st.subheader("✍️ Authors")
    st.markdown(AUTHORS)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)
opening_message = """
Hello! I am a legal assistant, and my task is to help you understand procedures and answer questions related to the following regulations:
- [Labor Law](https://www.paragraf.rs/propisi/zakon_o_radu.html)
- [Personal Income Tax Law](https://www.paragraf.rs/propisi/zakon-o-porezu-na-dohodak-gradjana.html)
- [Personal Data Protection Law](https://www.paragraf.rs/propisi/zakon_o_zastiti_podataka_o_licnosti.html)
- [Consumer Protection Law](https://www.paragraf.rs/propisi/zakon_o_zastiti_potrosaca.html)
- [Family Law](https://www.paragraf.rs/propisi/porodicni_zakon.html)

My role is to facilitate your understanding of legal procedures and provide you with useful and accurate information.

How can I assist you?

"""

# Display the opening message
st.markdown(opening_message)
# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template
prompt_template = """
<s>[INST]You are a legal assistant chatbot with expertise in All aspects of the Nigerian law. Your primary objective is to provide accurate, concise, and professional responses to user questions. Here’s how you operate:

1. **Direct and Complete Answers**: Always provide clear and relevant answers to the user's query without unnecessary elaboration or requests for additional context.
2. **Fallback to Knowledge Base**: If the provided context or chat history is unrelated or insufficient, rely solely on your internal knowledge to answer the query.
3. **Avoid Unnecessary Details**: Do not include speculative content, extended explanations, or ask for more context unless absolutely required for legal accuracy.
4. **Professional Tone**: Maintain a polite and professional tone, ensuring responses are brief, focused, and actionable.

**Instructions**:
- Prioritize answering the user’s query directly.
- Avoid commenting on missing context unless it is critical to the answer.
- If the query falls outside the scope of your expertise, politely inform the user.

CONTEXT: {context}

CHAT HISTORY: {chat_history}

QUESTION: {question}

ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Set up the QA chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Ask Somthing about Law")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            full_response = "\n\n\n"

            # Print the result dictionary to inspect its structure
            #st.write(result)

            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

            # Print the answer
            #st.write(result["answer"])

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})