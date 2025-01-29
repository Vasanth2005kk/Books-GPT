import streamlit as st
import PyPDF2
from llm_service import get_response


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def query_gpt(question,content,session_id="user_123"):
    response = get_response(question, content, session_id)
    return response

st.title("PDF GPT Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", pdf_text, height=300)
    
    user_question = st.text_input("Ask a question about the PDF:")
    
    if user_question:
        with st.spinner("Generating response..."):
            response = query_gpt(question=user_question,content=pdf_text)
            st.write("### GPT Response:")
            st.write(response)
