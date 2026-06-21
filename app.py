import os
import shutil
from io import BytesIO

import qrcode
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================================================
# CONFIG
# =========================================================
PERSIST_FILE = "vector_store_cache.json"
ADMIN_PASSWORD = "harjit123"
FREE_MESSAGE_LIMIT = 3

# UPI payment details
YOUR_UPI_ID = "harjeet.pahwa@oksbi"
MERCHANT_NAME = "Harjit Homeopathy"
PREMIUM_PRICE_INR = "99"      # launch offer (regular 199)

# Unlock code: give this ONLY to customers who have actually paid.
# Change it anytime (e.g. monthly) to keep it private.
UNLOCK_CODE = "HARJIT99"
# Your WhatsApp number for customers to send their payment screenshot.
WHATSAPP_NUMBER = "8800138095"   # <-- replace with your number, keep 91 prefix

# =========================================================
# OPENAI KEY
# =========================================================
if "OPENAI_API_KEY" in os.environ:
    pass
elif hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Homeopathic Assistant", page_icon="🌿", layout="centered")

# =========================================================
# STYLED LEAF HEADER
# =========================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .leaf-header {
        background: linear-gradient(90deg, #22c55e 0%, #10b981 50%, #0d9488 100%);
        padding: 22px 26px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(16,185,129,0.25);
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .leaf-header h1 {
        color: white; margin: 0; font-size: 28px; font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .leaf-header p { color: #ecfdf5; margin: 2px 0 0 0; font-size: 14px; }
    .leaf-icon { font-size: 40px; line-height: 1; }
</style>

<div class="leaf-header">
    <div class="leaf-icon">🌿</div>
    <div>
        <h1>Homeopathic Assistant</h1>
        <p>Natural remedy guidance from your knowledge base</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0
if "has_paid" not in st.session_state:
    st.session_state.has_paid = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# VECTOR STORE
# =========================================================
def load_vector_store():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    if os.path.exists(PERSIST_FILE) and os.path.getsize(PERSIST_FILE) > 0:
        try:
            embeddings = OpenAIEmbeddings()
            return InMemoryVectorStore.load(PERSIST_FILE, embeddings)
        except Exception:
            return None
    return None

if "vector_store" not in st.session_state or st.session_state.vector_store is None:
    st.session_state.vector_store = load_vector_store()

# =========================================================
# ADMIN SIDEBAR (only you can process PDFs)
# =========================================================
with st.sidebar:
    st.header("Admin Settings")
    password = st.text_input("Admin Password", type="password")

    if password == ADMIN_PASSWORD:
        st.success("Authenticated!")
        uploaded_file = st.file_uploader("Upload Knowledge Base (PDF)", type="pdf")

        if uploaded_file is not None:
            progress = st.progress(0)
            status = st.empty()

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            status.text("Reading PDF...")
            progress.progress(20)
            loader = PyPDFLoader("temp.pdf")
            raw_documents = loader.load()

            status.text("Splitting text...")
            progress.progress(45)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=100, length_function=len
            )
            chunks = splitter.split_documents(raw_documents)

            status.text("Generating AI vectors (this takes a moment)...")
            progress.progress(70)
            embeddings = OpenAIEmbeddings()
            new_store = InMemoryVectorStore.from_documents(chunks, embeddings)

            status.text("Saving knowledge base...")
            progress.progress(90)
            new_store.dump(PERSIST_FILE)
            st.session_state.vector_store = new_store

            os.remove("temp.pdf")
            progress.progress(100)
            status.text("Done!")
            st.success("Knowledge base updated!")
            st.rerun()
    elif password:
        st.error("Incorrect password.")

# =========================================================
# RAG CHAIN
# =========================================================
def answer_question(store, question):
    retriever = store.as_retriever(search_kwargs={"k": 8})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    system_prompt = (
        "You are an expert Homeopathic Assistant. Use the retrieved context from the "
        "uploaded book to help the user find remedies for their symptoms.\n\n"
        "Instructions:\n"
        "1. Identify the symptom(s) in the user's message, even if briefly stated.\n"
        "2. Name 3-4 specific remedies from the context and explain what makes each "
        "relevant (modalities, triggers, specific symptom pictures).\n"
        "3. Ask 2-3 follow-up questions to narrow down the best remedy.\n"
        "If the context truly contains nothing about the symptom, say so.\n\n"
        "Retrieved Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"question": question})

# =========================================================
# PAYWALL QR
# =========================================================
def upi_qr_png():
    clean_name = MERCHANT_NAME.replace(" ", "%20")
    upi_link = f"upi://pay?pa={YOUR_UPI_ID}&pn={clean_name}&am={PREMIUM_PRICE_INR}&cu=INR"
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(upi_link)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return upi_link, buf.getvalue()

# =========================================================
# MAIN CHAT INTERFACE
# =========================================================
if st.session_state.vector_store is None:
    st.info("The chatbot is currently offline. Open Admin Settings in the left sidebar "
            "to upload a PDF and initialize the knowledge base.")
else:
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(
            {"role": "assistant",
             "content": "Hello! Describe your symptoms clearly to discover tailored remedy matches."}
        )

    # render history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    reached_limit = (
        st.session_state.user_message_count >= FREE_MESSAGE_LIMIT
        and not st.session_state.has_paid
    )

    if reached_limit:
        st.warning(f"You have reached your limit of {FREE_MESSAGE_LIMIT} free messages.")

        upi_link, qr_bytes = upi_qr_png()

        st.markdown(
            f"""
            <div style="background:#f8fafc; padding:22px; border-radius:14px;
                        border:2px solid #e2e8f0; text-align:center; margin-bottom:16px;">
                <h3 style="color:#0f766e; margin-bottom:4px;">Unlock Unlimited Consultations 💎</h3>
                <p style="color:#475569; font-size:15px;">Launch offer:
                   <span style="text-decoration:line-through; color:#94a3b8;">₹199</span>
                   <b style="color:#0f766e;">₹{PREMIUM_PRICE_INR}</b> — one-time, for unrestricted access.</p>
                <a href="{upi_link}" style="background:#10b981; color:white; padding:12px 24px;
                   text-decoration:none; border-radius:8px; font-weight:bold; display:inline-block;">
                   Pay via Google Pay / PhonePe / UPI</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(qr_bytes, caption="Scan with any UPI app to pay", use_container_width=True)

            st.markdown(
                f"""
                <div style="text-align:center; color:#475569; font-size:14px; margin:8px 0 14px 0;">
                    After paying <b>₹{PREMIUM_PRICE_INR}</b>, send your payment screenshot on
                    WhatsApp to <b>{WHATSAPP_NUMBER}</b> and you'll receive your
                    <b>unlock code</b>. Enter it below to get unlimited access.
                </div>
                """,
                unsafe_allow_html=True,
            )

            code = st.text_input("Enter your unlock code", type="password")
            if st.button("Unlock Access"):
                if code.strip().upper() == UNLOCK_CODE.upper():
                    st.session_state.has_paid = True
                    st.success("Premium access unlocked! Enjoy unlimited consultations.")
                    st.rerun()
                else:
                    st.error("Incorrect code. Please check the code sent to you after payment.")
    else:
        if user_query := st.chat_input("Type your symptoms or question..."):
            st.chat_message("user").write(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.user_message_count += 1

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        answer = answer_question(st.session_state.vector_store, user_query)
                    except Exception as e:
                        answer = f"Error generating answer: {e}"
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
