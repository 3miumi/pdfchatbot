import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import time

from collections import defaultdict

# Set up the OpenAI client with API key
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Assuming you store your API key in Streamlit's secrets
# client = OpenAI(api_key=OPENAI_API_KEY)
def show():
    if "current_page" not in st.session_state or st.session_state.current_page != "ask_questions":
        st.session_state.current_page = "ask_questions"
        st.session_state.messages = []
    st.title("💬 PDF Question Page")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False



    def clear_state():
        st.session_state.messages = []
        st.session_state.uploader_key += 1  # Increment key to reset file uploader

    def stream_response_by_word(response):
        for word in response.split():
            yield word + " "
            time.sleep(0.1)
    st.markdown("""
        <p style='color: gray;'>Here you can reset the conversation by click button below.</p>
        """, unsafe_allow_html=True)
    if st.button('Clear Conversation'):
        clear_state()
        st.session_state.file_processed = False



    st.sidebar.markdown("""
       ## File Uploader
       Below is the file uploader where you can upload your PDF file to analyze. Make sure the file is in PDF format.
       """)
    with st.sidebar:
        with st.form(key='pdf_form', clear_on_submit=True):
            file = st.file_uploader("Upload a PDF file", type="pdf",
                                    key=f"file_uploader_{st.session_state.uploader_key}")
            submit_button = st.form_submit_button("Submit")

        # Handling API Key input
        st.header("Configuration")
        if 'user_api_key' not in st.session_state:
            st.session_state.user_api_key = ""
        st.session_state.user_api_key = st.text_input("Enter your API Key", value=st.session_state.user_api_key,
                                                      type="password")

    # Check if API key is set
    if st.session_state.user_api_key:
        OPENAI_API_KEY = st.session_state.user_api_key
        st.sidebar.success("API Key saved successfully!")
        client = OpenAI(api_key=st.session_state.user_api_key)
        # Continue with the rest of your application logic
        # Now use OPENAI_API_KEY for your API calls
    else:
        st.sidebar.error("Please enter your API key.")
        return



    if submit_button and client:
        st.session_state.file_processed = True  # Mark that a file has been processed
        if file:
            try:
                pdf_reader = PdfReader(file)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if not text.strip():
                    raise ValueError(
                        "No text could be extracted from the uploaded PDF. Please upload a PDF with selectable text.")

                text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
                chunks = text_splitter.split_text(text)
                if not chunks:
                    raise ValueError("Text was extracted but could not be processed into manageable chunks.")

                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.retriever = st.session_state.vector_store.as_retriever()
                st.success("PDF processed successfully. Ready to answer questions based on the PDF content.")
            except Exception as e:
                st.error(str(e))

        else:
            st.error("Please upload a file before submitting.")

    elif submit_button and not client:
        st.sidebar.error("Please enter your API key before submission")

    if not st.session_state.file_processed and submit_button:
        st.error("Please upload a file before submitting.")
        st.session_state.uploader_key += 1  # Reset uploader key here as well

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if the question can be answered from the PDF
        if 'vector_store' in st.session_state:
            with st.chat_message("assistant"):
                # match = st.session_state.vector_store.similarity_search(user_input)
                # llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000, model_name="gpt-4")
                # chain = load_qa_chain(llm, chain_type="stuff")  # Customize chain_type based on your needs
                # response = chain.run(input_documents=match, question=user_input)
                # response = stream_response_by_word(response)
                # response = st.write_stream(response)


                # Use the retriever to fetch relevant documents based on the user query

                llm = ChatOpenAI(openai_api_key=st.session_state.user_api_key, temperature=0, max_tokens=1000, model_name="gpt-4")
                retrieval_qa = RetrievalQA.from_chain_type(llm, chain_type="stuff",retriever =  st.session_state.retriever)
                response = retrieval_qa.invoke( user_input)["result"]
                print(response)
                response = stream_response_by_word(response)
                response = st.write_stream(response)
        else:
            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                )
                response = st.write_stream(response)

        st.session_state.messages.append({"role": "assistant", "content": response})




if __name__ == "__main__":

    show()