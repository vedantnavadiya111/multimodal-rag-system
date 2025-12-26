import streamlit as st
import logging
import os
import shutil
import pdfplumber
import ollama
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="QueryCore",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def extract_model_names() -> Tuple[str, ...]:
    logger.info("Extracting model names")
    models_info = ollama.list()
    model_names = tuple(m.model for m in models_info.models if m.model != "llama2")
    logger.info(f"Extracted model names: {model_names}")
    return model_names

@st.cache_resource
def get_embeddings():
    logger.info("Using Ollama Embeddings")
    return OllamaEmbeddings(model="nomic-embed-text")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error at get_text_chunks function: {e}")
        chunks = []
    return chunks

def get_vector_store(text_chunks):
    vector_store = None
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logger.error(f"Error at get_vector_store function: {e}")
    return vector_store

@st.cache_resource
def get_llm(selected_model: str):
    logger.info(f"Getting LLM: {selected_model}")
    return ChatOllama(model=selected_model, temperature=0.1)

def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = get_llm(selected_model)
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question as detailed as possible from the provided context only. 
    Do not generate a factual answer if the information is not available. 
    If you do not know the answer, respond with "I don’t know the answer as not sufficient information is provided in the PDF."
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Set up the chain with retriever and LLM
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get the response from the chain
    response = chain.invoke(question)

    # Check if the retrieved context is relevant or not
    if "I don’t know the answer" in response or not response.strip():
        return "I don’t know the answer as not sufficient information is provided in the PDF."

    logger.info("Question processed and response generated")
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages

def delete_vector_db() -> None:
    logger.info("Deleting vector DB")
    st.session_state.pop("pdf_pages", None)
    st.session_state.pop("file_upload", None)
    st.session_state.pop("vector_db", None)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        logger.info("FAISS index deleted")
    st.success("Collection and temporary files deleted successfully.")
    logger.info("Vector DB and related session state cleared")
    st.rerun()

def main():
    st.markdown(
        """
        <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 30px 0 10px 0; border-radius: 0 0 20px 20px; box-shadow: 0 4px 12px rgba(30,60,114,0.15); margin-bottom: 30px;'>
            <h1 style='color: #fff; text-align: center; font-size: 2.8rem; font-weight: 700; letter-spacing: 2px; margin-bottom: 0;'>QueryCore</h1>
            <p style='color: #e0e0e0; text-align: center; font-size: 1.2rem; margin-top: 0;'>Your Professional PDF Question Answering Platform</p>
            <p style='color: #ffd700; text-align: center; font-size: 1rem; margin-top: 10px;'>Created by Vedant Navadiya</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    available_models = extract_model_names()

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    with col1:
        st.markdown(
            """
            <div style='background: #f5f7fa; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(30,60,114,0.08);'>
                <h3 style='color: #1e3c72; font-weight: 600; margin-bottom: 10px;'>Upload PDF Files</h3>
                <p style='color: #444; font-size: 1rem;'>Select one or more PDF files to process and ask questions about their content.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True
        )
        col_buttons = st.columns([1, 1])

        with col_buttons[0]:
            submit_button = st.button("Process PDF(s)", key="submit_process", use_container_width=True)
        with col_buttons[1]:
            delete_collection = st.button("🗑️ Delete Collection", type="secondary", use_container_width=True)

        if submit_button and pdf_docs:
            with st.spinner("Processing your PDF(s)..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state["vector_db"] = get_vector_store(text_chunks)
                st.success("PDF(s) processed and ready for queries!", icon="✅")

            # Assuming single file upload for display, extract and store pages
            st.session_state["pdf_pages"] = extract_all_pages_as_images(pdf_docs[0])

        if delete_collection:
            delete_vector_db()

        # Centralized PDF viewer display
        if st.session_state.get("pdf_pages"):
            zoom_level = st.slider(
                "Zoom Level", min_value=100, max_value=1000, value=700, step=50
            )
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    with col2:
        st.markdown(
            """
            <div style='background: #f5f7fa; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(30,60,114,0.08);'>
                <h3 style='color: #1e3c72; font-weight: 600; margin-bottom: 10px;'>Ask Questions</h3>
                <p style='color: #444; font-size: 1rem;'>Type your question below to get answers based on your uploaded PDF(s).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if available_models:
            selected_model = st.selectbox(
                "Choose a local model for answering your queries:", available_models
            )

        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "🧑"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        prompt = st.chat_input("Type your question here...")
        if prompt:
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="🧑").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":blue[Processing your question...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                        else:
                            response = "Please upload and process a PDF file first."
                            st.warning(response)

            except Exception as e:
                st.error(e, icon="⚠️")
                logger.error(f"Error processing prompt: {e}")

if __name__ == "__main__":
    main()
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            text-align: center;
            padding: 10px 0;
            font-size: 1rem;
            letter-spacing: 1px;
            box-shadow: 0 -2px 8px rgba(30,60,114,0.08);
            z-index: 100;
        }
        </style>
        <div class='footer'>QueryCore | Created by Vedant Navadiya</div>
        """,
        unsafe_allow_html=True
    )