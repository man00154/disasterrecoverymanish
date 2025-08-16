# app.py
import os
import json
from typing import List
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangGraph (very simple visualization)
import networkx as nx
from pyvis.network import Network

# -------- Config ----------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

st.set_page_config(page_title="DC Disaster Recovery Assistant", layout="wide")
st.title("MANISH - GenAI-powered Disaster Recovery Assistant for Data Centers")
st.markdown("""
- Real-time guidance for disaster scenarios.
- Generates **custom recovery plans**.
- Visualize plan dependencies via simple LangGraph.
""")

# ---------------- Minimal RAG ----------------
class TinyRAG:
    def __init__(self):
        self.docs: List[str] = []
        self.tfidf = None
        self.vectors = None

    def index_texts(self, texts: List[str]):
        self.docs = texts
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
        self.vectors = self.tfidf.fit_transform(self.docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.vectors is None or not self.docs:
            return []
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [self.docs[i] for i in idxs if sims[i] > 0]

# ---------------- Gemini / fallback ----------------
def generate_with_gemini(prompt: str) -> str:
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        return local_fallback(prompt)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {"prompt": {"text": prompt}, "temperature": 0.2, "maxOutputTokens": 800}
    try:
        r = requests.post(API_URL, headers=headers, json=body, timeout=20)
        r.raise_for_status()
        resp = r.json()
        if "candidates" in resp and resp["candidates"]:
            return resp["candidates"][0].get("content","") or resp["candidates"][0].get("display","")
        if "output" in resp and isinstance(resp["output"], list):
            return " ".join([str(x.get("content","")) if isinstance(x, dict) else str(x) for x in resp["output"]])
        if "text" in resp:
            return resp["text"]
        return json.dumps(resp, indent=2)[:4000]
    except Exception as e:
        st.warning(f"Gemini call failed: {e}. Using local fallback.")
        return local_fallback(prompt)

def local_fallback(prompt: str) -> str:
    return ("Local fallback disaster recovery plan:\n"
            "- Assess affected systems.\n"
            "- Activate backups/failover.\n"
            "- Notify stakeholders.\n"
            "- Execute recovery steps with priority.\n"
            "- Review and update DR plan post-incident.")

# ---------------- Plan generation ----------------
def generate_recovery_plan(rag: TinyRAG, query: str, context_notes: str = "") -> str:
    snippets = rag.retrieve(query, top_k=5)
    prompt_parts = [
        "You are an expert data center disaster recovery assistant.",
        "Provide a detailed, actionable recovery plan based on context and recent query."
    ]
    if snippets:
        prompt_parts.append("Context snippets:")
        prompt_parts.extend([f"- {s[:800]}" for s in snippets])
    if context_notes:
        prompt_parts.append("Organization notes:")
        prompt_parts.append(context_notes[:1000])
    prompt_parts.append(f"Query:\n{query}")
    prompt_parts.append("Produce a step-by-step disaster recovery plan.")
    prompt = "\n\n".join(prompt_parts)
    return generate_with_gemini(prompt)

# ---------------- Simple LangGraph Visualization ----------------
def visualize_plan(plan_steps: List[str]):
    G = nx.DiGraph()
    for i, step in enumerate(plan_steps):
        G.add_node(i, label=step[:50])
        if i > 0:
            G.add_edge(i-1, i)
    net = Network(height="400px", width="100%", directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.save_graph("dr_plan.html")
    return "dr_plan.html"

# ---------------- Streamlit UI ----------------
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_texts = st.file_uploader("Upload reference text files (optional)", type=["txt","md"], accept_multiple_files=True)
    paste_context = st.text_area("Or paste context snippets (one per line)", height=120)
    st.markdown("---")
    st.markdown("**Gemini API key** (optional): set `GEMINI_API_KEY` env var or in Streamlit secrets.")

# Process texts
texts = []
if uploaded_texts:
    for f in uploaded_texts:
        try:
            txt = f.read().decode("utf-8")
        except:
            try:
                txt = f.read().decode("latin-1")
            except:
                txt = ""
        if txt:
            texts.append(txt)

if paste_context:
    for line in paste_context.splitlines():
        if line.strip():
            texts.append(line.strip())

if not texts:
    texts.append("Default knowledge snippet: Follow DR best practices, ensure backups, failovers, and vendor contacts ready.")

rag = TinyRAG()
rag.index_texts(texts)
st.success(f"Indexed {len(texts)} knowledge snippets.")

st.header("Disaster Recovery Assistant")
query = st.text_area("Describe current disaster scenario / issue", height=150)

if st.button("Generate Recovery Plan") and query.strip():
    with st.spinner("Generating disaster recovery plan..."):
        plan_text = generate_recovery_plan(rag, query)
        st.subheader("Recommended Recovery Plan")
        st.write(plan_text)

        # Very simple LangGraph visualization
        steps = [s.strip() for s in plan_text.split("\n") if s.strip()]
        html_file = visualize_plan(steps)
        st.subheader("Plan Dependencies Visualization (LangGraph)")
        st.components.v1.html(open(html_file, 'r').read(), height=450, scrolling=True)
