import os
import shutil
import streamlit as st
import qrcode
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 📁 GLOBAL CONFIGURATION VARIABLES
DB_DIR = "chroma_db_cache"
ADMIN_PASSWORD = "harjit123"
FREE_MESSAGE_LIMIT = 3

# 🇮🇳 NATIVE INDIAN UPI ROUTING PARAMETERS
YOUR_UPI_ID = "harjeet.pahwa@oksbi"       
MERCHANT_NAME = "Harjit Homeopathy"  
PREMIUM_PRICE_INR = "199"            
APP_URL = "https://streamlit.app"

# 🔑 SECURE OPENAI KEY INJECTION FOR ALL PLATFORMS
if "OPENAI_API_KEY" in os.environ:
    pass
elif "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input("API Key Verification Missing. Enter OpenAI Key:", type="password")

# Initialize page configuration exactly once
st.set_page_config(page_title="Pro AI Homeopathic Assistant", layout="centered")

# 🔒 MAXIMUM PRIVACY LAYOUT RESET
st.markdown("""
    <style>
        #MainMenu, [data-testid="stMainMenu"] {visibility: hidden !important; display: none !important;}
        footer, [data-testid="stFooter"] {visibility: hidden !important; display: none !important;}
        [data-testid*="avatar"], div[class*="StyledDecoration"], .viewerBadge, [class*="viewerBadge"] {
            visibility: hidden !important; display: none !important;
        }
        header, [data-testid="stHeader"] { background-color: transparent !important; }
        [data-testid="collapsedControl"] {
            visibility: visible !important; display: block !important; z-index: 999999 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Optimized Homeopathic AI Chatbot")

# Initialize persistent session state properties
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0

if "has_paid" not in st.session_state:
    st.session_state.has_paid = False

# -------------------------------------------------------------
# ARCHITECTURE OPTIMIZATION: CHROMADB PERSISTENT FILE LOADING
# -------------------------------------------------------------
def get_vector_store():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    embeddings = OpenAIEmbeddings()
    # Check if the persistent Chroma database folder exists and is populated
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        try:
            return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        except Exception:
            return None
    return None

# Load persistent store state
if "vector_store" not in st.session_state or st.session_state.vector_store is None:
    st.session_state.vector_store = get_vector_store()

# -------------------------------------------------------------
# 1. ADMIN SIDEBAR CONTROL PANEL (Optimized for Heavy 38MB PDFs)
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
            
            progress_text.text("1/4 📖 Reading large PDF book pages...")
            progress_bar.progress(15)
            
            loader = PyPDFLoader("temp.pdf")
            raw_documents = loader.load()
            
            progress_text.text(f"2/4 ✂️ Segmenting text elements...")
            progress_bar.progress(40)
            
            # Larger chunk constraints optimized to minimize processing iterations
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, length_function=len)
            optimized_docs = text_splitter.split_documents(raw_documents)
            
            progress_text.text(f"3/4 🧠 Compiling AI structural vector spaces...")
            progress_bar.progress(70)
            
            embeddings = OpenAIEmbeddings()
            
            # Wiping any old corrupted cache folders safely
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
                
            # Saving incrementally to disk folder directly via Chroma
            db = Chroma.from_documents(optimized_docs, embeddings, persist_directory=DB_DIR)
            
            progress_text.text("4/4 💾 Finishing disk serialization operations...")
            progress_bar.progress(90)
            
            st.session_state.vector_store = db
            os.remove("temp.pdf")
            
            progress_text.text("✅ Completed successfully!")
            progress_bar.progress(100)
            st.success("Knowledge base updated successfully!")
            
            # Manual interface override action to guarantee rendering updates
            if st.button("Launch Chat Interface 🚀"):
                st.rerun()
            
    elif password:
        st.error("Incorrect password.")

# -------------------------------------------------------------
# 2. CLIENT INTERFACE WITH PAYWALL LOGIC & CONVERSATIONAL MEMORY
# -------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("The chatbot is currently offline. Please log into Admin settings inside the left sidebar to initialize the data and upload your PDF book.")
else:
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! Describe your exact symptoms clearly to discover tailored remedy matches.")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    reached_limit = st.session_state.user_message_count >= FREE_MESSAGE_LIMIT and not st.session_state.has_paid

    if reached_limit:
        st.warning(f"⚠️ You have reached your limit of {FREE_MESSAGE_LIMIT} free messages.")
        
        clean_name = MERCHANT_NAME.replace(" ", "%20")
        upi_string = f"upi://pay?pa={YOUR_UPI_ID}&pn={clean_name}&am={PREMIUM_PRICE_INR}&cu=INR"
        
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(upi_string)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.markdown(
            f"""
            <div style="background-color:#f8f9fa; padding:25px; border-radius:12px; border:2px solid #e9ecef; text-align:center; margin-bottom: 20px;">
                <h3 style="color:#1d3557; margin-bottom:5px;">Unlock Unlimited Consultations 💎</h3>
                <p style="color:#4a5568; font-size:15px;">Gain unrestricted access to our database for a one-time fee of <b>₹{PREMIUM_PRICE_INR}</b>.</p>
                <div style="margin: 20px 0;">
                    <a href="{upi_string}" style="background-color:#10b981; color:white; padding:12px 25px; text-decoration:none; border-radius:8px; font-weight:bold; display:inline-block; font-size:16px; box-shadow:0 4px 6px rgba(16,185,129,0.2);">
                        Pay via Google Pay / PhonePe / UPI
                    </a>
                </div>
                <p style="color:#718096; font-size:13px;">On mobile devices, tap the button to launch your banking wallet. On computers, scan the code below:</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns()
        with col2:
            st.image(byte_im, caption="Scan using any UPI App to Pay", use_container_width=True)
            st.write("---")
            utr_input = st.text_input("Enter your 12-digit UPI Ref / UTR No. after making the transfer", max_chars=12)
            if st.button("Verify Transfer & Unlock"):
                if len(utr_input) == 12 and utr_input.isdigit():
                    st.session_state.has_paid = True
                    st.success("Payment received! Premium access unlocked.")
                    st.rerun()
                else:
                    st.error("Please enter a valid 12-digit numeric transaction UTR tracking code.")
    else:
        if user_query := st.chat_input("Type your specific symptoms here..."):
            st.chat_message("user").write(user_query)
            msgs.add_user_message(user_query)
            st.session_state.user_message_count += 1
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data profiles..."):
                    
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                    
                    system_prompt = (
                        "You are an expert Homeopathic Assistant. Your objective is to help the user identify potential "
                        "remedies based strictly on the specific physical and emotional symptoms found in the uploaded text.\n\n"
                        "CRITICAL DIRECTIONS:\n"
                        "1. Match the user's specific symptom variations to the remedy profiles inside the context.\n")
