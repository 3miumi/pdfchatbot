import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity



def classify_query_gpt3(query,client):
    prompt = (
        "Given a user query about a collection of PDF documents, classify the intent of the query into the following categories:\n"
        "1. Similarity Check: The user wants to understand how documents are similar or different.For example, if a user asks 'What are the common themes between these two reports?', this should be classified as a 'Similarity Check' because the user wants to know how the documents relate or differ in content.\n"
        "2. General Inquiry: The user asks general questions about the content or nature of the documents. For example, if a user asks 'What types of documents are these?' or 'what topics are those documents', it should be classified as 'General Inquiry' as it's about the nature of the documents in general.\n"
        "3. Other: Any other types of inquiries.\n"
        f"Query: '{query}'\n"
        "Classification:"
    )
    response =client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=["\n"],
        temperature=0.0  # Lower temperature ensures more consistent outputs
    )
    return response.choices[0].text.strip()
def get_all_embeddings(vector_store):
    # Retrieve all embeddings from the FAISS vector store
    nb_vectors = vector_store.index.ntotal
    embeddings = vector_store.index.reconstruct_n(0, nb_vectors)
    return embeddings

def get_chunk(doc_id, chunk_index):
    return st.session_state.all_chunks.get(doc_id, {}).get(chunk_index, None)
def calculate_similarity(embeddings, metadata):
    similarity_matrix = cosine_similarity(embeddings)
    num_docs = len(metadata)
    cross_doc_similarity = np.zeros_like(similarity_matrix)
    for i in range(num_docs):
        for j in range(num_docs):
            if metadata[i]['doc_id'] != metadata[j]['doc_id']:
                cross_doc_similarity[i][j] = similarity_matrix[i][j]
            else:
                cross_doc_similarity[i][j] = -1
    return cross_doc_similarity




def find_least_similar(similarity_matrix, metadata, top_n=2):
    least_similar = []
    num_docs = len(metadata)
    for i in range(len(similarity_matrix)):
        # Find indices of least similar chunks that belong to different documents
        sorted_indices = np.argsort(similarity_matrix[i])  # This sorts in ascending order (least similar first)
        current_differences = []
        for idx in sorted_indices:
            if metadata[i]['doc_id'] != metadata[idx]['doc_id'] and similarity_matrix[i][idx] < 0.8:  # Threshold can be adjusted
                current_differences.append((idx, similarity_matrix[i][idx]))
                if len(current_differences) >= top_n:
                    break

        if current_differences:
            least_similar.append((metadata[i], current_differences))
    return least_similar
def find_top_similarities(similarity_matrix, metadata, top_n=2):
    top_similarities = []
    for i in range(len(similarity_matrix)):
        top_indices = np.argsort(similarity_matrix[i])[::-1][:top_n + 1]
        current_similarities = [(j, similarity_matrix[i][j]) for j in top_indices if i != j and similarity_matrix[i][j] != -1]
        if current_similarities:
            top_similarities.append((metadata[i], current_similarities))
    return top_similarities


def generate_comprehensive_explanation(top_similarities, least_similarities, metadata,client):
    if top_similarities:
        prompt = "Explain the similarities between the following sections of documents,no need for difference analyze and summarize them :\n"
        count_sim = 0
        average_length = st.session_state.get('average_chunk_length', 1000)  # Default to 1000 if not set

        # Dynamic max_items based on average length
        if average_length < 500:
            max_items = 10
        elif average_length < 1000:
            max_items = 5
        else:
            max_items = 3
        print(max_items)
        for doc_meta, sims in top_similarities:
            for sim_idx, sim_score in sims:
                if count_sim >= max_items:
                    break
                sim_meta = metadata[sim_idx]
                c_index = doc_meta['chunk_index']
                id = doc_meta["doc_id"]
                chunk_1= (get_chunk(id,c_index)["text"])

                c_index_2 = sim_meta['chunk_index']
                id_2 = sim_meta["doc_id"]
                chunk_2= (get_chunk(id_2,c_index_2)["text"])

                prompt += f"-  {chunk_1} of '{doc_meta['title']}' and  {chunk_2} of '{sim_meta['title']}')\n"
                count_sim += 1


        messages = st.session_state.messages.copy()  # Copy existing conversation history
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.5,
            stream=True
        )
        return response
    return "No significant similarities were found."


def clear_state():
    st.session_state.messages = []
    st.session_state.all_chunks = defaultdict(dict)
    st.session_state.all_metadata = []

    st.session_state.uploader_key += 1  # Increment key to reset file uploader


def stream_response_by_word(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.1)




if "current_page" not in st.session_state or st.session_state.current_page != "document_comparison":
    st.session_state.current_page = "document_comparison"
    st.session_state.messages = []

st.title("💬 Compare Your Documents")
st.info("""
    This page supports the comparison of two PDF documents and queries regarding multiple PDFs.
    If you wish to focus on querying within a single PDF, please use the 'Ask Question' page.
""")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False




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

        files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=f"file_uploader_{st.session_state.uploader_key}")

        submit_button = st.form_submit_button("Submit")
    st.header("Configuration")
    if 'user_api_key' not in st.session_state:
        st.session_state.user_api_key = ""
    st.session_state.user_api_key = st.text_input("Enter your API Key", value=st.session_state.user_api_key,
                                                    type="password")
    if st.session_state.user_api_key:
        OPENAI_API_KEY = st.session_state.user_api_key
        st.sidebar.success("API Key saved successfully!")
        # Continue with the rest of your application logic
        # Now use OPENAI_API_KEY for your API calls
    else:
        st.sidebar.error("Please enter your API key.")
        

if submit_button:

    st.session_state.file_processed = True  # Mark that a file has been processed
    if files:
        all_chunks = []
        st.session_state.all_chunks=defaultdict(dict)

        st.session_state.all_metadata = []
        total_length = 0
        num_chunks = 0
        try:
            for doc_id, doc in enumerate(files, start=1):
                pdf_reader = PdfReader(doc)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if not text.strip():
                    raise ValueError(
                        "No text could be extracted from the uploaded PDF. Please upload a PDF with selectable text.")

                text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
                chunks = text_splitter.split_text(text)
                if not chunks:
                    raise ValueError("Text was extracted but could not be processed into manageable chunks.")
                for chunk_index, chunk in enumerate(chunks):
                    chunk_length = len(chunk)
                    total_length += chunk_length
                    num_chunks += 1
                    # Create metadata for each chunk
                    metadata = {
                        "doc_id": doc_id,
                        "title": doc.name,  # Using file name as title
                        "chunk_index": chunk_index
                    }
                    st.session_state.all_chunks[doc_id][chunk_index] = {
                        "text": chunk,
                        "metadata": metadata
                    }
                    all_chunks.append(chunk)
                    st.session_state.all_metadata.append(metadata)
            if num_chunks > 0:
                st.session_state.average_chunk_length = total_length / num_chunks
            else:
                st.session_state.average_chunk_length = 0
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            st.session_state.vector_store = FAISS.from_texts(all_chunks, embeddings,metadatas=st.session_state.all_metadata )
            st.session_state.retriever = st.session_state.vector_store.as_retriever()
            st.success("PDF processed successfully. Ready to answer questions based on the PDF content.")
        except Exception as e:
            st.error(str(e))

    else:
        st.error("Please upload a file before submitting.")

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
    client = OpenAI(api_key=OPENAI_API_KEY)
    classification = classify_query_gpt3(user_input,client)
    if classification == "Similarity Check" and 'vector_store' in st.session_state:
        with st.chat_message("assistant"):
            embeddings = get_all_embeddings(st.session_state.vector_store)
            similarity_matrix = calculate_similarity(embeddings, st.session_state.all_metadata )
            top_similarities = find_top_similarities(similarity_matrix, st.session_state.all_metadata )
            lesat_similarities = find_least_similar(similarity_matrix, st.session_state.all_metadata)
            response = generate_comprehensive_explanation(top_similarities, lesat_similarities,st.session_state.all_metadata , client)
            # response = stream_response_by_word(explanation)
            response = st.write_stream(response)
            print("Similarity search")
    # Check if the question can be answered from the PDF

    elif classification == "General Inquiry" and 'vector_store' in st.session_state:
        with st.chat_message("assistant"):
            match = st.session_state.vector_store.similarity_search(user_input)
            print("vector_store search")
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000, model_name="gpt-4")
            chain = load_qa_chain(llm, chain_type="stuff")  # Customize chain_type based on your needs
            response = chain.run(input_documents=match, question=user_input)
            response = stream_response_by_word(response)
            response = st.write_stream(response)
            # retrieval_qa = RetrievalQA.from_chain_type(llm, chain_type="stuff",
            #                                            retriever=st.session_state.retriever)
            # response = retrieval_qa.invoke(user_input)["result"]
            #
            # response = stream_response_by_word(response)
            # response = st.write_stream(response)s
    else:
        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            print("General search")
            response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


    print( st.session_state.messages)
