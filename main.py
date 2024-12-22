import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import asyncio

# Page configuration
st.set_page_config(page_title="Webpage Searching Engine", layout="wide", page_icon="🌐")

# Sidebar for input and info
with st.sidebar:
    st.image("http://surl.li/fpkhbj", width=100)
    st.title("Interactive Chat on Website 🌐")
    st.caption("Interact with a webpage seamlessly through this app, which uses a local Llama3 model and Retrieval-Augmented Generation (RAG).")

    # Input URL
    webpage_url = st.text_input("Enter the URL of the webpage you want to interact with:")
    st.write(f"You entered: {webpage_url}")

# Main chat interface in the center
st.markdown("""
    <h1 style='text-align: center;'>Webpage Chatbot</h1>
""", unsafe_allow_html=True)

# Clean text function
def clean_text(text):
    return " ".join(text.split()).strip()

# Cache vector store
@st.cache_resource
def cache_vector_store(webpage_url):
    loader = WebBaseLoader(webpage_url)
    documents = loader.load()

    # Clean documents without converting them to strings
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Optimized chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Async function to interact with Ollama
async def ollama_llm_async(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(
        model="llama3.1",
        messages=[{'role': "user", 'content': formatted_prompt}]
    )
    return response["message"]["content"]

# RAG setup
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def ragchain(question, retriever):
    retriever_docs = retriever.invoke(question)
    formatted_context = combine_docs(retriever_docs)
    response = await ollama_llm_async(question, formatted_context)
    return response

# Main function for Streamlit
if webpage_url:
    # Retrieve cached vector store
    vectorstore = cache_vector_store(webpage_url)

    # Set up the retriever
    retriever = vectorstore.as_retriever()

    # Chat interface in the center
    st.markdown("""
        <div style='display: flex; justify-content: center;'>
            <h2>Chatbot Interface</h2>
        </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()

    # Chat history feature
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    with chat_container:
        question = st.text_input("Enter your question:", key="question_input")
        if question:
            # Run the async function when the question is entered
            response = asyncio.run(ragchain(question, retriever))
            st.session_state['chat_history'].append((question, response))

            st.text_area("Answer:", value=response, height=200, max_chars=None, key="response_area")

            # Add 'copy to clipboard' button
            st.button("Copy to Clipboard", on_click=st.code, args=(response,))

        # Display chat history
        st.markdown("**Chat History:**")
        for q, a in st.session_state['chat_history']:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
