import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

ADMIN_PASSWORD = "harjit123" # Change this!

st.set_page_config(page_title="AI Customer Assistant", layout="centered")
st.title("🤖 Customer Support Chatbot")
st.write("Welcome! Ask any question about our services or documentation below.")

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
    if user_query := st.chat_input("Type your question here..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the global state vector retriever
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                
                system_prompt = (
                    "You are a helpful customer service assistant. Use the following pieces of retrieved "
                    "context to answer the question. If you don't know the answer, say that you don't know.\n\n"
                    "Context:\n{context}"
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
