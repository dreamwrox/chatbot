import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 📁 GLOBAL CONFIGURATION VARIABLES
PERSIST_FILE = "vector_store_cache.json"
ADMIN_PASSWORD = "harjit123"
FREE_MESSAGE_LIMIT = 3
PAYMENT_LINK = "https://stripe.com" 

st.set_page_config(page_title="AI Homeopathic Assistant", layout="centered")
st.title("🌿 Homeopathic AI Chatbot")

# Initialize user message counter
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0

# Track if the user has paid
if "has_paid" not in st.session_state:
    st.session_state.has_paid = False

# -------------------------------------------------------------
# PERMANENT FILE CACHING FOR PDF DATA
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(PERSIST_FILE):
        try:
            return InMemoryVectorStore.load(PERSIST_FILE, embeddings)
        except Exception:
            return None
    return None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_cached_vector_store()

# -------------------------------------------------------------
# 1. ADMIN SIDEBAR (Generates the JSON file automatically)
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
                
                loader = PyPDFLoader("temp.pdf")
                raw_documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                optimized_docs = text_splitter.split_documents(raw_documents)
                
                embeddings = OpenAIEmbeddings()
                vector_store = InMemoryVectorStore.from_documents(optimized_docs, embeddings)
                
                # Cleanup old legacy directory blocks if they exist
                if os.path.exists("vector_store_cache") and os.path.isdir("vector_store_cache"):
                    shutil.rmtree("vector_store_cache")
                
                # Auto-generate the fresh JSON file
                vector_store.dump(PERSIST_FILE)
                
                st.session_state.vector_store = vector_store
                os.remove("temp.pdf")
                st.success("Knowledge base updated successfully!")
                st.rerun()
    elif password:
        st.error("Incorrect password.")

# -------------------------------------------------------------
# 2. CLIENT INTERFACE WITH PAYWALL LOGIC
# -------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("The chatbot is currently offline. Please log into Admin settings to initialize the medical data.")
else:
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! Describe your exact symptoms clearly to discover tailored remedy matches.")

    # Render history UI
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Check if user hit the limit
    reached_limit = st.session_state.user_message_count >= FREE_MESSAGE_LIMIT and not st.session_state.has_paid

    if reached_limit:
        st.warning(f"⚠️ You have reached your limit of {FREE_MESSAGE_LIMIT} free messages.")
        st.markdown(
            f"""
            <div style="background-color:#fff3cd; padding:20px; border-radius:10px; border:1px solid #ffeeba; text-align:center;">
                <h3>Unlock Unlimited Consultations 💎</h3>
                <p>Gain unrestricted access to our entire homeopathic database and get unlimited answers.</p>
                <a href="{PAYMENT_LINK}" target="_blank" style="background-color:#28a745; color:white; padding:10px 20px; text-decoration:none; border-radius:5px; font-weight:bold; display:inline-block; margin-top:10px;">
                    Unlock Premium Access Now
                </a>
            </div>
            <br>
            """, 
            unsafe_allowed_html=True
        )
        
        if st.button("Simulate Successful Payment (For Developer Testing)"):
            st.session_state.has_paid = True
            st.success("Premium access unlocked!")
            st.rerun()
            
    else:
        if user_query := st.chat_input("Type your specific symptoms here..."):
            st.chat_message("user").write(user_query)
            st.session_state.user_message_count += 1
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data profiles..."):
                    
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 6})
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                    
                    system_prompt = (
                        "You are an expert Homeopathic Assistant. Your objective is to help the user identify potential "
                        "remedies based strictly on the specific physical and emotional symptoms found in the uploaded text.\n\n"
                        "CRITICAL DIRECTIONS:\n"
                        "1. Match the user's specific symptom variations to the remedy profiles inside the context.\n"
                        "2. Avoid generic summaries. Explicitly name 3-4 distinct remedies from the text and point out "
                        "exactly what makes each remedy relevant (modalities, specific pain types, triggers).\n"
                        "3. Always ask the client 2 to 3 clear, specific questions to narrow down the correct remedy profile.\n\n"
                        "Retrieved Homeopathic Context:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}"),
                    ])
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {
                            "context": retriever | format_docs,
                            "question": RunnablePassthrough(),
                            "history": lambda x: msgs.messages
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(user_query)
                    st.write(response)
                    
                    msgs.add_user_message(user_query)
                    msgs.add_ai_message(response)
                    st.rerun()
