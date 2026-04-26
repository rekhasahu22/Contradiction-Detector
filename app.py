import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
from utils import detect_contradiction, highlight_words

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Contradiction Detector", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### 🔍 About Project
This system detects relationships between sentences using NLP.

### ⚙️ Techniques Used
- TF-IDF Vectorization  
- Logistic Regression  
- Text Similarity Analysis  

### 📊 Output Types
- ❌ Contradiction  
- ✅ Entailment  
- ⚪ Neutral  

### 🚀 Features
- Sentence Analysis  
- Batch Processing  
- PDF Analysis  
- Keyword Highlighting  
- Analytics Dashboard  

### 🎯 Use Cases
- Fake News Detection  
- Chatbots  
- Document Analysis  
""")

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center;color:#4FC3F7'>
🤖 AI Contradiction Detector
</h1>
<p style='text-align:center;color:gray'>
Analyze relationship between two sentences
</p>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- INPUT ----------------
st.subheader("✍️ Enter Sentences")

col1, col2 = st.columns(2)

with col1:
    s1 = st.text_area("Sentence 1")

with col2:
    s2 = st.text_area("Sentence 2")

col3, col4, col5 = st.columns(3)

analyze = col3.button("🔍 Analyze")
example = col4.button("⚡ Example")
clear = col5.button("🧹 Clear History")

if example:
    s1 = "A man is running"
    s2 = "A man is sleeping"

if clear:
    st.session_state.history = []

# ---------------- HIGHLIGHT FUNCTION ----------------
def highlight(text, words, color):
    for w in words:
        text = text.replace(w, f"<span style='color:{color}; font-weight:bold'>{w}</span>")
    return text

# ---------------- ANALYZE ----------------
if analyze and s1 and s2:
    result, score = detect_contradiction(s1, s2)

    # RESULT
    if "Contradiction" in result:
        st.error(result)
    elif "Entailment" in result:
        st.success(result)
    else:
        st.warning(result)

    # CONFIDENCE
    st.info(f"Confidence Score: {round(score*100,2)}%")
    st.progress(score)

    # HIGHLIGHT
    st.markdown("### 🔍 Highlighted Sentences")

    diff1, diff2 = highlight_words(s1, s2)

    colh1, colh2 = st.columns(2)

    colh1.markdown(highlight(s1, diff1, "red"), unsafe_allow_html=True)
    colh2.markdown(highlight(s2, diff2, "orange"), unsafe_allow_html=True)

    # SAVE HISTORY
    if (s1, s2, result, score) not in st.session_state.history:
        st.session_state.history.append((s1, s2, result, score))

# ---------------- TXT FILE UPLOAD ----------------
st.markdown("---")
st.subheader("📄 Upload TXT File")

uploaded_file = st.file_uploader("Upload .txt file (use '||')", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    lines = content.split("\n")

    st.subheader("📊 File Results")

    for line in lines:
        if "||" in line:
            s1_file, s2_file = line.split("||")

            s1_file = s1_file.strip()
            s2_file = s2_file.strip()

            if s1_file and s2_file:
                st.markdown("---")
                st.markdown(f"🧑 **{s1_file}**")
                st.markdown(f"🧑 **{s2_file}**")

                result, score = detect_contradiction(s1_file, s2_file)

                st.markdown(f"🤖 **{result} ({round(score*100,2)}%)**")

                if (s1_file, s2_file, result, score) not in st.session_state.history:
                    st.session_state.history.append((s1_file, s2_file, result, score))

# ---------------- PDF UPLOAD (NEW FEATURE) ----------------
st.markdown("---")
st.subheader("📄 Upload PDF & Analyze")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    st.subheader("📑 Extracted Text Preview")
    st.text(text[:500])

    sentences = text.split(".")

    st.subheader("📊 PDF Analysis Results")

    for i in range(len(sentences)-1):
        s1_pdf = sentences[i].strip()
        s2_pdf = sentences[i+1].strip()

        if s1_pdf and s2_pdf:
            result, score = detect_contradiction(s1_pdf, s2_pdf)

            st.markdown("---")
            st.markdown(f"🧑 **{s1_pdf}**")
            st.markdown(f"🧑 **{s2_pdf}**")
            st.markdown(f"🤖 **{result} ({round(score*100,2)}%)**")

# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("💬 History")

for item in st.session_state.history[::-1]:
    st.markdown(f"🧑 **{item[0]}**")
    st.markdown(f"🧑 **{item[1]}**")
    st.markdown(f"🤖 **{item[2]} ({round(item[3]*100,2)}%)**")
    st.write("---")

# ---------------- DOWNLOAD ----------------
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history, columns=["S1","S2","Result","Score"])
    st.download_button("📥 Download Results", df.to_csv(index=False), "results.csv")

# ---------------- ANALYTICS ----------------
if st.session_state.history:
    st.subheader("📊 Analytics")

    labels = {"Contradiction":0, "Entailment":0, "Neutral":0}

    for item in st.session_state.history:
        if "Contradiction" in item[2]:
            labels["Contradiction"] += 1
        elif "Entailment" in item[2]:
            labels["Entailment"] += 1
        else:
            labels["Neutral"] += 1

    colA, colB = st.columns(2)

    fig1, ax1 = plt.subplots()
    ax1.bar(labels.keys(), labels.values())
    colA.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.pie(labels.values(), labels=labels.keys(), autopct='%1.1f%%')
    colB.pyplot(fig2)