import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "chroma_db"
ADMIN_PASSWORD = "harjit1234" # Change this!

st.set_page_config(page_title="AI Customer Assistant", layout="centered")
st.title("🤖 Customer Support Chatbot")
st.write("Welcome! Ask any question about our services or documentation below.")

# -------------------------------------------------------------
# 1. ADMIN SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.header("Admin Settings")
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("Authenticated!")
        uploaded_file = st.file_uploader("Upload New Knowledge Base (PDF)", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Processing PDF and updating knowledge base..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load_and_split()
                embeddings = OpenAIEmbeddings()
                
                # Create and save Chroma database
                db = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
                
                os.remove("temp.pdf")
                st.success("Knowledge base updated successfully!")
    elif password:
        st.error("Incorrect password.")

# -------------------------------------------------------------
# 2. CLIENT CHAT INTERFACE
# -------------------------------------------------------------
if not os.path.exists(DB_DIR):
    st.info("The chatbot is currently offline. Please contact the administrator to upload documentation.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Type your question here..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embeddings = OpenAIEmbeddings()
                db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 3})
                
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
                
                # Modern LangChain Chain Layout (Bypasses classic chains entirely)
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
