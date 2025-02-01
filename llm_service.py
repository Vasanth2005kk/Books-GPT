from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.globals import set_llm_cache
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.cache import SQLiteCache  # Updated import
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from sqlalchemy import create_engine
from flask import session
import os
import sqlite3

# Check if database exists, if not, create it
DB_path = os.getcwd() + "/langchain.db"
if not os.path.exists(DB_path):
    conn = sqlite3.connect(DB_path)
    conn.close()
    print(f"Database '{DB_path}' created successfully!")

# Set the LLM cache
set_llm_cache(SQLiteCache(database_path=DB_path))


# Define the LLM model
llm = OllamaLLM(model="llama3.2:3b",base_url="http://localhost:11434")

# Define the embedding model for retrieval
ollama_emb = OllamaEmbeddings(model="llama3.2:3b")

# Store for session management
store = {}

# Contextualizing prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "Answer the following question from the context below"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    {context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Initialize chat history
chat_history = []


def get_session_history(session_id) -> BaseChatMessageHistory:
    engine = create_engine(f"sqlite:///{DB_path}")  # ✅ Correct SQLAlchemy connection
    return SQLChatMessageHistory(session_id, connection=engine)



def get_response(question, content, session_id):
    """
    This function processes the user input question and content to retrieve
    the relevant answer by utilizing history-aware retrieval chains.
    """
    global chat_history
    
    # Split the content into smaller chunks for efficient processing
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_text(content)

    # Create a vector store with FAISS and use it for retrieval
    vector = FAISS.from_texts(documents, ollama_emb)
    retriever = vector.as_retriever()

    # Set up the retrieval chain with history awareness
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create a conversational RAG chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Invoke the chain and get the answer
    ai_msg_1 = conversational_rag_chain.invoke({"input": question, "chat_history": chat_history}, config={
        "configurable": {"session_id": session_id}
    })

    # Update the chat history with the latest messages
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=ai_msg_1["answer"]),
        ]
    )

    # Print and return the model's answer
    # print(ai_msg_1['answer'])
    return ai_msg_1['answer']

if __name__ =="__main__":
    session_id = "user_123"
    content = "-"
    question = input("Enter the Questions:")
    response = get_response(question, content, session_id)
    print("Answer:", response)
