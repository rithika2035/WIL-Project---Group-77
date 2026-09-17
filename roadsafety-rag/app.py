"""
VIC Road Safety RAG Assistant
Professional Streamlit UI

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from retrieval import retrieve, detect_stage
from generation import generate_answer, DISTANCE_THRESHOLD


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VIC Road Safety Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       MAIN PAGE
       ========================= */

    .stApp {
        background-color: #0f172a;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       TEXT
       ========================= */

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #cbd5e1;
    }


    /* =========================
       HERO
       ========================= */

    .hero-box {
        background: linear-gradient(
            135deg,
            #172554,
            #1d4ed8
        );

        border-radius: 22px;
        padding: 2.4rem 2.6rem;
        margin-bottom: 2rem;

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.25);
    }

    .hero-box h1 {
        color: white !important;
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
    }

    .hero-box p {
        color: #dbeafe !important;
        font-size: 1rem;
        line-height: 1.6;
    }


    /* =========================
       QUESTION BOX
       ========================= */

    textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
    }

    textarea::placeholder {
        color: #94a3b8 !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }


    /* =========================
       CARDS
       ========================= */

    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1rem;
    }

    .card-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .card-text {
        color: #cbd5e1;
        line-height: 1.65;
    }


    /* =========================
       SUCCESS / WARNING
       ========================= */

    .success-box {
        background-color: #052e1b;
        border: 1px solid #166534;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: #86efac;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .warning-box {
        background-color: #431407;
        border: 1px solid #9a3412;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: #fdba74;
        font-weight: 600;
        margin-bottom: 1rem;
    }


    /* =========================
       EVIDENCE
       ========================= */

    .evidence {
        background-color: #172033;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
    }

    .evidence-title {
        color: #f8fafc;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .evidence-meta {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-bottom: 0.7rem;
    }

    .evidence-text {
        color: #cbd5e1;
        line-height: 1.6;
        font-size: 0.9rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }


    /* =========================
       METRICS
       ========================= */

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }


    /* =========================
       EXPANDER
       ========================= */

    div[data-testid="stExpander"] {
        background-color: #172033;
        border: 1px solid #334155;
        border-radius: 12px;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer-text {
        color: #64748b;
        text-align: center;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #334155;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ RAG Settings")

    st.caption(
        "Configure retrieval behaviour and inspect how the "
        "assistant finds supporting road-safety information."
    )

    st.markdown("### Retrieval")

    top_k = st.slider(
        "Evidence chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of passages retrieved from ChromaDB.",
    )

    filter_stage = st.checkbox(
        "Driver-stage filtering",
        value=True,
        help=(
            "Detects learner, P1, P2 or full licence context "
            "and filters retrieved passages accordingly."
        ),
    )

    st.divider()

    st.markdown("### 🧠 System")

    st.markdown(
        """
        **Embedding search**

        Sentence Transformers

        **Vector database**

        ChromaDB

        **Generation model**

        Ollama · llama3.2:3b

        **Guardrail**

        Retrieval confidence threshold
        """
    )

    st.divider()

    st.markdown("### 📚 Knowledge Base")

    st.caption(
        "The assistant uses indexed Victorian road-safety "
        "passages to generate evidence-grounded answers."
    )

    st.caption(
        f"Confidence threshold: {DISTANCE_THRESHOLD:.3f}"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-box">

    <h1>🛡️ VIC Road Safety Assistant</h1>

    <p>
    Evidence-grounded question answering for Victorian
    road rules and driver conditions.
    </p>

    <p>
    Ask a question and the system retrieves relevant
    road-safety evidence before generating a response.
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# QUESTION SECTION
# ============================================================

st.markdown("## Ask a road-safety question")

st.caption(
    "Choose an example or enter your own question."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

example_questions = [
    "Can a P1 driver use a hands-free phone?",
    "What are the alcohol restrictions for P1 drivers?",
    "What are the speed restrictions for learner drivers?",
]

columns = st.columns(3)

for i, example in enumerate(example_questions):

    with columns[i]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state["question"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Question",
    value=st.session_state.get("question", ""),
    placeholder=(
        "Example: Can a P1 driver use a hands-free phone "
        "while driving?"
    ),
    height=110,
    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

ask = st.button(
    "🔎  Find Answer",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN RAG PIPELINE
# ============================================================

if ask:

    if not question.strip():

        st.warning(
            "Please enter a road-safety question first."
        )

        st.stop()

    st.session_state["question"] = question

    # --------------------------------------------------------
    # Detect driver stage
    # --------------------------------------------------------

    detected_stage = (
        detect_stage(question)
        if filter_stage
        else None
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    with st.spinner(
        "Searching the road-safety knowledge base..."
    ):

        hits = retrieve(
            question,
            top_k=top_k,
            filter_stage=filter_stage,
        )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    with st.spinner(
        "Generating an evidence-grounded answer..."
    ):

        result = generate_answer(
            question,
            hits,
        )


    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown("## Answer")

    if result["used_fallback"]:

        with st.container(border=True):
            st.markdown(
                """
                <div class="warning-box">
                ⚠️ Insufficient evidence
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.warning(result["answer"])

            st.caption(
            "The confidence guardrail prevented the language "
            "model from generating an unsupported answer."
            
        )

    else:

        st.markdown(
            """
            <div class="success-box">
            ✓ Evidence found · Answer generated from retrieved passages
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### 🛡️ Road Safety Answer")
            st.markdown(result["answer"])
            st.divider()


    # ========================================================
    # RETRIEVAL SUMMARY
    # ========================================================

    if hits:

        st.markdown("## Retrieval Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Evidence chunks",
                len(hits),
            )

        with col2:

            st.metric(
                "Top distance",
                f"{hits[0]['distance']:.3f}",
            )

        with col3:

            st.metric(
                "Driver stage",
                detected_stage or "Not detected",
            )

        with col4:

            st.metric(
                "Guardrail",
                "Passed"
                if not result["used_fallback"]
                else "Fallback",
            )


    # ========================================================
    # SUPPORTING EVIDENCE
    # ========================================================

    st.markdown("## 📚 Supporting Evidence")

    st.caption(
        "Retrieved passages supplied to the generation stage."
    )

    for index, hit in enumerate(hits, start=1):

        metadata = hit.get("metadata", {})

        doc_id = metadata.get(
            "doc_id",
            metadata.get(
                "source",
                "Unknown source",
            ),
        )

        section = metadata.get(
            "section_title",
            "Section not specified",
        )

        page = metadata.get("page_number")

        stage = metadata.get(
            "driver_stage",
            "Not specified",
        )

        topic = metadata.get(
            "topic",
            "Not specified",
        )

        page_text = (
            f" · Page {page}"
            if page
            else ""
        )

        st.markdown(
            f"""
            <div class="evidence">

            <div class="evidence-title">
            Evidence {index} · {doc_id}
            </div>

            <div class="evidence-meta">
            📖 {section}{page_text}
            · 👤 {stage}
            · 🏷️ {topic}
            · Distance: {hit["distance"]:.3f}
            </div>

            <div class="evidence-text">
            {hit["text"]}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander("🔬 View RAG pipeline details"):

        st.markdown("### Query")

        st.code(
            question,
            language=None,
        )

        st.markdown("### Retrieval configuration")

        st.json(
            {
                "top_k": top_k,
                "driver_stage_filtering": filter_stage,
                "detected_stage": detected_stage,
                "confidence_threshold": DISTANCE_THRESHOLD,
            }
        )

        st.markdown("### Pipeline")

        st.markdown(
            """
            **1. Query**

            User enters a road-safety question.

            ↓

            **2. Embedding**

            Sentence Transformers converts the question
            into a semantic vector.

            ↓

            **3. Retrieval**

            ChromaDB searches the indexed road-safety passages.

            ↓

            **4. Driver-stage filtering**

            Learner, P1, P2 or full-licence context is detected
            when applicable.

            ↓

            **5. Confidence guardrail**

            The top retrieval distance is checked before
            calling the language model.

            ↓

            **6. Generation**

            Ollama generates an answer using the retrieved
            excerpts.

            ↓

            **7. Evidence**

            Retrieved passages are displayed for transparency.
            """
        )


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not ask:

    st.markdown("## 👋 Welcome")

    st.info(
        """
        Enter a Victorian road-safety question above to retrieve
        relevant evidence and generate an evidence-grounded answer.

        The system checks retrieval confidence before allowing
        the language model to generate a response.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
    VIC Road Safety RAG · ChromaDB · Sentence Transformers ·
    Ollama · Evidence-Grounded Question Answering
    </div>
    """,
    unsafe_allow_html=True,
)