import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Use a local directory to save files permanently on the server disk
PERSIST_DIR = "vector_store_cache.json"
ADMIN_PASSWORD = "harjit123"

st.set_page_config(page_title="Pro AI Homeopathic Assistant", layout="centered")
st.title("🌿 Optimized Homeopathic AI Chatbot")

# -------------------------------------------------------------
# ARCHITECTURE OPTIMIZATION 1: PERMANENT FILE CACHING
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_vector_store():
    """Loads vector index from disk if it exists, keeping data alive across reboots."""
    embeddings = OpenAIEmbeddings()
    # If saved previously, rebuild store instantly from saved directory
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        try:
            return InMemoryVectorStore.load(PERSIST_DIR, embeddings)
        except Exception:
            return None
    return None

# Load persistent store on app start
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_cached_vector_store()

# -------------------------------------------------------------
# ADMIN SIDEBAR CONTROL PANEL
# -------------------------------------------------------------
with st.sidebar:
    st.header("Admin Settings")
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("Authenticated!")
        uploaded_file = st.file_uploader("Upload New Knowledge Base (PDF)", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner("Optimizing text segments & building vector space..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest Document
                loader = PyPDFLoader("temp.pdf")
                raw_documents = loader.load()
                
                # OPTIMIZATION 2: SMART RECURSIVE CHUNKING
                # Instead of page cuts, cuts text by logical sentence/paragraph blocks with overlaps
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,      # Characters per block (~200 words)
                    chunk_overlap=200,    # 20% context carry-over to next block
                    length_function=len
                )
                optimized_docs = text_splitter.split_documents(raw_documents)
                
                # Build and Save Index Permanently
                embeddings = OpenAIEmbeddings()
                vector_store = InMemoryVectorStore.from_documents(optimized_docs, embeddings)
                
                vector_store.dump(PERSIST_FILE)

                
                st.session_state.vector_store = vector_store
                os.remove("temp.pdf")
                st.rerun()
    elif password:
        st.error("Incorrect password.")

# -------------------------------------------------------------
# CLIENT INTERFACE WITH CONVERSATIONAL MEMORY
# -------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("The chatbot is currently offline. Please log into Admin settings to initialize the medical data.")
else:
    # OPTIMIZATION 3: NATIVE SESSION CHAT HISTORY STORAGE
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! Describe your exact symptoms clearly to discover tailored remedy matches.")

    # Render complete stylized history UI
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_query := st.chat_input("Type your specific symptoms here..."):
        st.chat_message("user").write(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data profiles..."):
                
                # OPTIMIZATION 4: HIGH-RECALL SEMANTIC RETRIEVAL
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                
                # Prompt upgraded to handle conversational memory placeholders natively
                system_prompt = (
                    "You are an expert Homeopathic Assistant. Your objective is to help the user map out potential "
                    "remedies based strictly on the specific symptoms present within the provided reference text.\n\n"
                    "CRITICAL DIRECTIONS:\n"
                    "1. Match the user's specific symptom variations to the remedy profiles inside the context.\n"
                    "2. Avoid generic summaries. Explicitly name 3-4 distinct remedies from the text and point out "
                    "exactly what makes each remedy relevant (modalities, specific pain types, triggers).\n"
                    "3. Prompt the user with 2 targeted symptom extraction questions to further refine your analysis.\n"
                    "4. If historical context suggests a remedy was already proposed, build upon it instead of restarting.\n\n"
                    "Retrieved Homeopathic Context:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="history"), # Connects Conversation Memory
                    ("human", "{question}"),
                ])
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # Modern functional RAG chain logic orchestration
                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                        "history": lambda x: msgs.messages # Seamless injection of memory loop
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Run engine and update message memory store
                response = rag_chain.invoke(user_query)
                st.write(response)
                
                msgs.add_user_message(user_query)
                msgs.add_ai_message(response)
