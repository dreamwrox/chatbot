import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

ADMIN_PASSWORD = "harjit123" # Keep or change your password here

st.set_page_config(page_title="AI Homeopathic Assistant", layout="centered")
st.title("🌿 Homeopathic Support Chatbot")
st.write("Describe your exact symptoms below to find matching remedies from the uploaded documentation.")

# Initialize the vector store in Streamlit's global app session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# -------------------------------------------------------------
# 1. ADMIN SIDEBAR (Hidden behind password)
# -------------------------------------------------------------
with st.sidebar:
    st.header("Admin Settings")
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("Authenticated!")
        uploaded_file = st.file_uploader("Upload New Knowledge Base (PDF)", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF and updating knowledge base..."):
                # Save temp file
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load, split, and embed using pure Python structures
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load_and_split()
                embeddings = OpenAIEmbeddings()
                
                # Instantiating the clean InMemory Vector Store
                vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
                st.session_state.vector_store = vector_store
                
                os.remove("temp.pdf")
                st.success("Knowledge base updated successfully!")
    elif password:
        st.error("Incorrect password.")

# -------------------------------------------------------------
# 2. CLIENT CHAT INTERFACE (Visible to everyone)
# -------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("The chatbot is currently offline. Please contact the administrator to upload documentation.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input chat loop
    if user_query := st.chat_input("Type your symptoms here (e.g., dry cough, worse at night)..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documentation..."):
                # FIX 1: Increased 'k' to 7 so the AI reads much more data from your PDF at once
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 7})
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2) # Low temperature ensures strict factual accuracy
                
                # FIX 2: Upgraded system prompt to demand specific remedy profiling and diagnostic follow-ups
                system_prompt = (
                    "You are an expert Homeopathic Assistant. Your goal is to help the user identify potential "
                    "remedies based strictly on the specific physical and emotional symptoms found in the uploaded text.\n\n"
                    
                    "CRITICAL DIRECTIONS:\n"
                    "1. DO NOT give a generic, high-level overview or a generic step-by-step workflow chart.\n"
                    "2. Scan the retrieved context below for concrete homeopathic remedies (e.g., Aconite, Belladonna, Pulsatilla, Bryonia) "
                    "that match the user's condition. List 3 or 4 possible options alongside their specific symptom modalities "
                    "(what makes it better/worse, type of discharge, mental state) as written in the text.\n"
                    "3. Always ask the client 2 to 3 clear, specific questions to narrow down the correct remedy profile "
                    "(e.g., 'Is your cough wet or dry?', 'Does movement make the pain worse?', 'Are your nasal discharges watery or thick?').\n"
                    "4. If the retrieved context does not contain specific remedies for the query, politely ask the user for "
                    "more specific symptoms to help match the text data.\n\n"
                    
                    "Retrieved Homeopathic Documentation Context:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}"),
                ])
                
                # Streamlined Chain Layout
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Get response
                answer = rag_chain.invoke(user_query)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
