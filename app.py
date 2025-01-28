import streamlit as st
from rag import RAG
import os
from langchain.llms import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter



legalRag = RAG(llm=OpenAI(temperature=0.7),vector_db_path='./data')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

st.title("Generative AI for Legal Advice")
tab1, tab2 = st.tabs(["Upload PDF", "Chat with RAG"])

with tab1:
    st.header("Upload a PDF Document")

    DATA_FOLDER = "./uploaded_pdfs"

    st.title("File State Activation")

    existing_files = os.listdir(DATA_FOLDER) if os.path.exists(DATA_FOLDER) else []
    if existing_files:
        uploaded_file = existing_files[0]  # Get the first file
        file_path = os.path.join(DATA_FOLDER, uploaded_file)
        st.success(f"The file '{uploaded_file}' is already in the data folder. State: Activated ✅")
        legalRag.indexing(f"./uploaded_pdfs/{uploaded_file}", text_splitter)
    else:
        st.info("Please upload a file to check its state.")
        
    new_uploaded_file = st.file_uploader("Upload a file")
    
    if new_uploaded_file is not None:
        for file in os.listdir(DATA_FOLDER):
            file_path = os.path.join(DATA_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.error(f"Error removing file {file}: {str(e)}")
    
        with open(os.path.join(DATA_FOLDER, new_uploaded_file.name), "wb") as f:
            f.write(new_uploaded_file.getbuffer())
        
        st.rerun()
        


with tab2:
    st.title("Simple chat")

 
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.empty()  

    def render_chat():
        with chat_container.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    prompt = st.chat_input("What is up?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.messages.append({"role": "assistant", "content":"thinking..."})
        render_chat()
        
        response = legalRag.generate(prompt)
        
        st.session_state.messages.pop(-1)
        st.session_state.messages.append({"role": "assistant", "content": response})
        render_chat()

        
