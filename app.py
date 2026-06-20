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

st.set_page_config(page_title="Pro AI Homeopathic Assistant", layout="centered")

# 🔒 PRIVACY OPTIMIZATION: Hides Streamlit's branding, footer, and GitHub repository links
ABSOLUTE ULTIMATE PRIVACY BLOCK: Erases platform badges, avatars, and overlays
st.markdown("""
    <style>
        /* Hides the global system toolbar header container layout */
        header, [data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
        
        /* Hides the standard system footer attribution mark */
        footer, [data-testid="stFooter"] {visibility: hidden !important; display: none !important;}
        
        /* Hides the default hamburger menu */
        #MainMenu, [data-testid="stMainMenu"] {visibility: hidden !important; display: none !important;}
        
        /* FORCE HIDE THE USER IDENTITY AVATAR & DEV HOOKS OBJECTS */
        iframe, [class*="viewerBadge"], [data-testid*="avatar"], div[class*="StyledDecoration"] {
            visibility: hidden !important;
            display: none !important;
        }
        
        /* Overrides the lower-right tracking buttons if they escape basic rules */
        div[style*="position: fixed; right:"], div[style*="bottom: 0px; right: 0px;"] {
            visibility: hidden !important;
            display: none !important;
        }
        
        /* Adjust page spacing container layouts cleanly */
        .block-container {padding-top: 1rem !important;}
    </style>
""", unsafe_allow_html=True)




# 📁 GLOBAL CONFIGURATION VARIABLES
PERSIST_FILE = "vector_store_cache.json"
ADMIN_PASSWORD = "harjit123"
FREE_MESSAGE_LIMIT = 3
# Paste your custom redirect-enabled Stripe Link here:
PAYMENT_LINK = "https://stripe.com" 
# Configure your production web address for WhatsApp sharing:
APP_URL = "https://streamlit.app"

st.set_page_config(page_title="Pro AI Homeopathic Assistant", layout="centered")
st.title("🌿 Optimized Homeopathic AI Chatbot")

# Initialize state trackers
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0

if "has_paid" not in st.session_state:
    st.session_state.has_paid = False

# -------------------------------------------------------------
# 🔄 AUTOMATED PAYMENT VERIFICATION VIA URL PARAMS
# -------------------------------------------------------------
query_params = st.query_params

# If Stripe redirects them back with ?status=success, auto-unlock premium status
if "status" in query_params and query_params["status"] == "success":
    st.session_state.has_paid = True
    # Clear URL params out cleanly so refreshing the page later resets state if needed
    st.query_params.clear()
    st.success("🎉 Payment verified successfully! Premium access unlocked.")

# -------------------------------------------------------------
# ARCHITECTURE OPTIMIZATION: PERMANENT FILE CACHING
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(PERSIST_FILE) and os.path.getsize(PERSIST_FILE) > 0:
        try:
            return InMemoryVectorStore.load(PERSIST_FILE, embeddings)
        except Exception:
            return None
    return None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_cached_vector_store()

# -------------------------------------------------------------
# 1. ADMIN SIDEBAR CONTROL PANEL
# -------------------------------------------------------------
with st.sidebar:
    st.header("Admin Settings")
    password = st.text_input("Enter Admin Password", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("Authenticated!")
        uploaded_file = st.file_uploader("Upload New Knowledge Base (PDF)", type="pdf")
        
        if uploaded_file is not None:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            progress_text.text("1/4 📖 Reading PDF pages...")
            progress_bar.progress(15)
            
            loader = PyPDFLoader("temp.pdf")
            raw_documents = loader.load()
            
            progress_text.text(f"2/4 ✂️ Splitting {len(raw_documents)} pages into text chunks...")
            progress_bar.progress(40)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      
                chunk_overlap=50,    
                length_function=len
            )
            optimized_docs = text_splitter.split_documents(raw_documents)
            
            progress_text.text(f"3/4 🧠 Generating AI vectors for {len(optimized_docs)} chunks...")
            progress_bar.progress(70)
            
            embeddings = OpenAIEmbeddings()
            vector_store = InMemoryVectorStore.from_documents(optimized_docs, embeddings)
            
            progress_text.text("4/4 💾 Saving database file to disk...")
            progress_bar.progress(90)
            
            if os.path.exists("vector_store_cache") and os.path.isdir("vector_store_cache"):
                shutil.rmtree("vector_store_cache")
            
            vector_store.dump(PERSIST_FILE)
            st.session_state.vector_store = vector_store
            
            os.remove("temp.pdf")
            progress_text.text("✅ Completed successfully!")
            progress_bar.progress(100)
            st.success("Knowledge base updated successfully!")
            st.rerun()
            
    elif password:
        st.error("Incorrect password.")

# -------------------------------------------------------------
# 2. CLIENT INTERFACE WITH PAYWALL LOGIC & CONVERSATIONAL MEMORY
# -------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("The chatbot is currently offline. Please log into Admin settings to initialize the data.")
else:
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! Describe your exact symptoms clearly to discover tailored remedy matches.")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Check if user hit their usage limits
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
            unsafe_allow_html=True
        )
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
                    
                    # FIXED: Properly terminated formatting block setup
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

# -------------------------------------------------------------
# 3. FLOATING WHATSAPP SHARE BUTTON
# -------------------------------------------------------------
# -------------------------------------------------------------
# 3. FLOATING WHATSAPP SHARE BUTTON (ZERO TRIPLE-QUOTES FIX)
# -------------------------------------------------------------
SHARE_TEXT = "Check out this amazing AI Homeopathic Assistant! Get answers from expert documentation instantly:"
encoded_message = "https://web.whatsapp.com/" + SHARE_TEXT.replace(" ", "%20") + "%20" + APP_URL

# Inline single-line styling injection to prevent f-string crashes
st.markdown("<style>.float-wa { position: fixed; width: 60px; height: 60px; bottom: 40px; right: 40px; background-color: #25d366; color: white !important; border-radius: 50px; text-align: center; font-size: 30px; box-shadow: 2px 2px 3px #999; z-index: 1000; display: flex; align-items: center; justify-content: center; text-decoration: none !important; transition: all 0.3s ease; } .float-wa:hover { background-color: #128C7E; transform: scale(1.1); }</style>", unsafe_allow_html=True)

# Directly render the actionable HTML link block
st.markdown('<a href="' + encoded_message + '" class="float-wa" target="_blank" title="Share this bot on WhatsApp">💬</a>', unsafe_allow_html=True)
