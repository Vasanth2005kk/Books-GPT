import streamlit as st
import PyPDF2
from llm_service import get_response
from models import list_ollama_models,suppourtFormats
import time

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def query_gpt(question,content,session_id="user_123"):
    response = get_response(question, content, session_id)
    return response

# Set the title of the app
st.title("PDF GPT Chatbot")

# Sidebar for file upload and model selection
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF file", type=suppourtFormats())
    if uploaded_file:
        st.radio("Your local machine models 👇".title(), list_ollama_models())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Container to display chat history
messages = st.container(height=600)

# Display chat history
for msg in st.session_state.messages:
    with messages.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Say something"):
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with messages.chat_message("user"):
        st.write(f"user :{prompt}")

    # Generate assistant response (dummy response for now)
    response = f"{query_gpt(question=prompt,content=extract_text_from_pdf(pdf_file=uploaded_file))}"
    
    # Function to simulate streaming data
    def stream_data():
        for word in response.split(" "):
            yield word + " "
            time.sleep(0.02)

    # Append assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant message with streaming effect
    with messages.chat_message("assistant"):
        with st.status("BOT :", expanded=True) as status:
            st.write_stream(stream_data())
            status.update(label="", state="complete")