# ----------------------------
# 🤖 RAG Chatbot Interface – Ask the Docs (All Features Enhanced)
# ----------------------------

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import openai
import requests
import json
import datetime
from sentence_transformers import SentenceTransformer
from io import StringIO

# ----------------------------
# 🔧 App Configuration
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 Ask the Docs – RAG Chatbot")

# Initialize Embedding Model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chat log storage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# 📥 Upload PDFs
# ----------------------------
st.sidebar.header("📥 Upload Documentation PDFs")
pdf_files = st.sidebar.file_uploader("Upload 3–5 PDFs", type="pdf", accept_multiple_files=True)

# ----------------------------
# 🧠 Model Selection
# ----------------------------
model_choice = st.sidebar.selectbox("🧠 Choose LLM Provider", ["Together AI", "OpenAI (GPT-4)"])
openai_key = st.sidebar.text_input("🔑 OpenAI Key", type="password")
together_key = st.sidebar.text_input("🔑 Together AI Key", type="password", value="9a500448c4b38072995a6c57d316a40797cb82a51908bd04844f7f5ec8d2c3c4")

# ----------------------------
# 📄 Process PDF Chunks
# ----------------------------
corpus_chunks = []
chunk_sources = []
full_text_map = {}

if pdf_files:
    for pdf_file in pdf_files:
        file_name = pdf_file.name
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text_full = ""
        for page in doc:
            text = page.get_text()
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            corpus_chunks.extend(chunks)
            chunk_sources.extend([file_name] * len(chunks))
            text_full += text + "\n"
        full_text_map[file_name] = text_full

    embeddings = embed_model.encode(corpus_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.success(f"✅ Loaded {len(corpus_chunks)} chunks from {len(pdf_files)} PDF(s).")

    # Optional: Summarize PDFs
    if st.sidebar.button("📝 Summarize All PDFs"):
        for file, content in full_text_map.items():
            summary_prompt = f"Summarize the document below:\n\n{content}\n\nSummary:"
            st.write(f"**📄 {file} Summary:**")
            try:
                if model_choice == "Together AI":
                    headers = {"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"}
                    payload = {"model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "messages": [
                        {"role": "system", "content": "You are a summarizer."},
                        {"role": "user", "content": summary_prompt}
                    ]}
                    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, data=json.dumps(payload))
                    summary = response.json()["choices"][0]["message"]["content"].strip()
                    st.info(summary)
                else:
                    openai.api_key = openai_key
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a summarizer."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    summary = response.choices[0].message.content.strip()
                    st.info(summary)
            except Exception as e:
                st.warning(f"Failed to summarize {file}: {e}")

# ----------------------------
# 💬 Chat Interface
# ----------------------------
if corpus_chunks:
    st.subheader("🤖 Ask Your PDFs Anything")
    query = st.text_input("Enter your question")

    if query:
        query_embed = embed_model.encode([query])
        D, I = index.search(np.array(query_embed), k=5)
        top_chunks = [corpus_chunks[i] for i in I[0]]
        top_sources = [chunk_sources[i] for i in I[0]]

        context = "\n\n".join(top_chunks)
        prompt = f"Answer the following based on the documentation context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        with st.spinner("Generating answer..."):
            try:
                if model_choice == "Together AI":
                    headers = {"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"}
                    payload = {"model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "messages": [
                        {"role": "system", "content": "You are a helpful documentation assistant."},
                        {"role": "user", "content": prompt}
                    ]}
                    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, data=json.dumps(payload))
                    answer = response.json()["choices"][0]["message"]["content"].strip()
                else:
                    openai.api_key = openai_key
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful documentation assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.choices[0].message.content.strip()

                st.markdown(f"### 📥 Answer\n{answer}")

                # Save chat log
                st.session_state.chat_history.append({
                    "timestamp": str(datetime.datetime.now()),
                    "question": query,
                    "answer": answer,
                    "sources": top_sources
                })

            except Exception as e:
                st.error(f"❌ Failed to get response: {e}")

    # Export chat history
    if st.session_state.chat_history:
        st.subheader("🧾 Chat History")
        for log in st.session_state.chat_history:
            st.markdown(f"**{log['timestamp']}**\n- **Q:** {log['question']}\n- **A:** {log['answer']}\n- **📄 Sources:** {', '.join(log['sources'])}")

        csv_export = "timestamp,question,answer,sources\n" + "\n".join(
            [f"{log['timestamp']},{log['question']},{log['answer'].replace(',', ' ')},{'; '.join(log['sources'])}" for log in st.session_state.chat_history]
        )
        st.download_button("📤 Download Chat Log", data=csv_export.encode(), file_name="chat_history.csv", mime="text/csv")
else:
    st.info("Upload at least one PDF to get started.")