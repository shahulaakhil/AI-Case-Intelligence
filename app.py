# ============================================================
# AI CASE INTELLIGENCE
# Evidence-Grounded Analysis of Indian Legal Judgments
# ============================================================

import streamlit as st
import fitz
import faiss
import re
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Case Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 8


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .source-box {
        background-color: #f5f5f5;
        padding: 9px 14px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 14px;
    }

    .answer-box {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
    }

    .point {
        margin-bottom: 8px;
        line-height: 1.6;
    }

    .footer {
        color: #777;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ AI Case Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Evidence-Grounded Analysis of Indian Legal Judgments'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL
    )

    model.eval()

    return tokenizer, model


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("Loading AI models..."):

    embedding_model = load_embedding_model()

    tokenizer, llm_model = load_llm()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for i in range(len(document)):

        page = document[i]

        text = page.get_text("text")

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if text:

            pages.append(
                {
                    "page": i + 1,
                    "text": text
                }
            )

    document.close()

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append(
                    {
                        "id": len(chunks),
                        "page": page_number,
                        "text": chunk_text
                    }
                )

            if end >= len(text):
                break

            start += (
                chunk_size - overlap
            )

    return chunks


# ============================================================
# FAISS INDEX
# ============================================================

def create_faiss_index(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    stopwords = {
        "the",
        "and",
        "was",
        "were",
        "are",
        "is",
        "of",
        "to",
        "in",
        "for",
        "on",
        "a",
        "an",
        "whether",
        "what",
        "who",
        "which",
        "with",
        "from",
        "that",
        "this",
        "has",
        "have",
        "had",
        "by"
    }

    return set(
        word
        for word in words
        if word not in stopwords
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_evidence(
    query,
    chunks,
    index,
    top_k=TOP_K,
    special_pages=None
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    search_k = min(
        len(chunks),
        max(top_k * 3, 20)
    )

    semantic_scores, indices = index.search(
        query_embedding,
        search_k
    )

    query_words = tokenize_text(
        query
    )

    candidates = []

    for score, idx in zip(
        semantic_scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        chunk = chunks[idx]

        text_words = tokenize_text(
            chunk["text"]
        )

        if query_words:

            keyword_score = len(
                query_words.intersection(
                    text_words
                )
            ) / len(query_words)

        else:

            keyword_score = 0.0

        final_score = (
            0.75 * float(score)
            +
            0.25 * keyword_score
        )

        if special_pages:

            if chunk["page"] in special_pages:

                final_score += 0.10

        candidates.append(
            {
                "page": chunk["page"],
                "text": chunk["text"],
                "score": final_score
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    results = []

    seen = set()

    for item in candidates:

        key = (
            item["page"],
            item["text"][:180]
        )

        if key in seen:
            continue

        seen.add(key)

        results.append(
            item
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    evidence,
    max_chars=6500
):

    blocks = []

    current_length = 0

    for item in evidence:

        block = (
            f"[SOURCE PAGE {item['page']}]\n"
            f"{item['text']}\n\n"
        )

        if (
            current_length + len(block)
            > max_chars
        ):
            break

        blocks.append(
            block
        )

        current_length += len(block)

    return "".join(
        blocks
    )


# ============================================================
# LLM GENERATION
# ============================================================

def generate_answer(
    prompt,
    max_tokens=450
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.no_grad():

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_answer(answer):

    # Remove common prompt leakage
    answer = re.sub(
        r"(?im)^TASK:.*$",
        "",
        answer
    )

    answer = re.sub(
        r"(?im)^ANSWER:.*$",
        "",
        answer
    )

    answer = re.sub(
        r"(?im)^QUESTION:.*$",
        "",
        answer
    )

    answer = re.sub(
        r"(?im)^JUDGMENT TEXT:.*$",
        "",
        answer
    )

    answer = re.sub(
        r"(?im)^CONTEXT:.*$",
        "",
        answer
    )

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    return answer.strip()


# ============================================================
# CONVERT EVERYTHING TO POINTS
# ============================================================

def convert_to_points(text):

    text = clean_answer(
        text
    )

    if not text:

        return []

    # Remove markdown bullets/numbers first
    text = re.sub(
        r"(?m)^\s*[-•*]\s*",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*\d+[\.\)]\s*",
        "",
        text
    )

    # Split lines
    lines = text.splitlines()

    points = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Ignore obvious prompt fragments
        if line.lower() in {
            "task",
            "answer",
            "context",
            "judgment evidence"
        }:
            continue

        points.append(
            line
        )

    # If model produced one large paragraph,
    # split into sentences.
    if len(points) == 1:

        single = points[0]

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            single
        )

        if len(sentences) > 1:

            points = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

    return points


# ============================================================
# SECTION CONFIGURATION
# ============================================================

SECTION_CONFIG = {

    "📄 Case Summary": {

        "query": """
        case facts accused appellant respondent
        prosecution allegations charges
        trial court high court supreme court
        final decision judgment outcome
        """,

        "instruction": """
        Summarize the case in 5 to 7 clear points.

        Each point should cover one of:
        - case identity
        - parties
        - prosecution allegations
        - charges
        - important procedural history
        - court's decision
        - final result

        Every point must contain useful case information.
        """
    },


    "👥 Key Persons": {

        "query": """
        appellant accused respondent complainant
        victim deceased witness PW-1 PW-2 PW-3
        investigating officer police officer
        prosecution witness defence witness
        judge names roles
        """,

        "instruction": """
        Identify 4 to 10 important persons.

        Write one point for each person.

        Format:

        Name — Role — Relevant involvement.

        Include only people explicitly supported by the judgment.
        """
    },


    "📅 Timeline": {

        "query": """
        dated date incident occurrence
        complaint FIR e-FIR registered
        investigation arrest apprehension
        recovery seizure
        charge sheet trial
        judgment high court appeal
        supreme court
        """,

        "instruction": """
        Create a chronological timeline.

        Write one point for every important event.

        Format:

        Date/Event — What happened.

        Include:
        - incident
        - complaint/FIR
        - investigation
        - arrest/apprehension
        - recovery/seizure
        - charge sheet
        - trial
        - lower court judgment
        - appeal
        - final judgment

        Do not invent dates.
        """
    },


    "⚖️ Legal Issues": {

        "query": """
        questions for consideration
        issues legal issue
        Section 379 IPC
        Section 411 IPC
        burden proof
        offence
        conviction acquittal
        court determine
        """,

        "instruction": """
        Identify the main legal issues.

        Write each issue as a separate point.

        For every point include:
        - issue/question
        - relevant legal provision
        - what the court had to determine

        Only mention provisions explicitly discussed
        in the judgment.
        """
    },


    "🔎 Evidence & Findings": {

        "query": """
        evidence testimony witness statement
        recovery seizure
        prosecution evidence
        defence evidence
        police investigation
        confession
        admission
        court found
        court held
        credibility
        """,

        "instruction": """
        Explain the important evidence and findings
        in 5 to 8 separate points.

        Each point should follow:

        Evidence — Court finding.

        Clearly distinguish:
        - what was alleged
        - what a witness stated
        - what evidence was produced
        - what the court actually found
        """
    },


    "⚠️ Contradictions / Gaps": {

        "query": """
        contradiction inconsistency discrepancy
        omission missing evidence
        prosecution failed
        investigation lapse
        procedural defect
        unreliable witness
        absence of evidence
        recovery doubt
        """,

        "instruction": """
        Identify specific contradictions or evidentiary gaps.

        Write each as a separate point.

        Focus on:
        - contradictory testimony
        - missing evidence
        - unreliable evidence
        - investigation shortcomings
        - failure to establish required facts
        - procedural gaps

        Do not give generic legal principles.
        """
    },


    "🧠 Judgment Reasoning": {

        "query": """
        court held
        court observed
        reasoning
        therefore
        hence
        prosecution failed
        beyond reasonable doubt
        burden of proof
        evidence insufficient
        conviction cannot be sustained
        acquittal
        """,

        "instruction": """
        Explain the court's reasoning in 5 to 8 points.

        Follow this logical structure:

        Evidence → Legal requirement → Court finding → Consequence.

        Explain why the court accepted or rejected important
        evidence and how that led to the decision.

        Do not give your own opinion.
        """
    },


    "🏛️ Final Outcome": {

        "query": """
        ORDER
        final order
        operative portion
        appeal allowed
        appeal dismissed
        appeal partly allowed
        conviction
        acquitted
        acquittal
        conviction set aside
        sentence
        bail bond
        discharged
        disposed
        """,

        "instruction": """
        State the final outcome in 3 to 6 separate points.

        Focus specifically on the operative/final order.

        Include:
        - appeal result
        - conviction/acquittal
        - charges
        - sentence
        - bail/release/discharge
        - other final directions

        Do not use general reasoning as the final outcome.
        """
    }
}


# ============================================================
# SPECIAL PAGE PRIORITIES
# ============================================================

def get_special_pages(
    section_name,
    total_pages
):

    pages = set()

    # Final order is normally at the end
    if section_name == "🏛️ Final Outcome":

        for page in range(
            max(1, total_pages - 3),
            total_pages + 1
        ):

            pages.add(page)

    # Timeline generally requires early procedural pages
    elif section_name == "📅 Timeline":

        for page in range(
            1,
            min(total_pages, 8) + 1
        ):

            pages.add(page)

    # Key persons commonly appear early
    elif section_name == "👥 Key Persons":

        for page in range(
            1,
            min(total_pages, 8) + 1
        ):

            pages.add(page)

    return pages


# ============================================================
# ANALYZE SECTION
# ============================================================

def analyze_section(
    section_name,
    config,
    chunks,
    index,
    total_pages
):

    special_pages = get_special_pages(
        section_name,
        total_pages
    )

    evidence = retrieve_evidence(
        config["query"],
        chunks,
        index,
        top_k=TOP_K,
        special_pages=special_pages
    )

    context = build_context(
        evidence
    )

    if not context:

        return {
            "points": [
                "Not clearly established from the provided judgment."
            ],
            "pages": []
        }

    prompt = f"""
You are analyzing an Indian court judgment.

SECTION:
{section_name}

TASK:
{config["instruction"]}

STRICT RULES:

- Use ONLY the supplied judgment evidence.
- Do not invent facts.
- Do not guess.
- Do not use outside information.
- Do not give legal advice.
- Do not give your personal opinion.
- Distinguish allegations from findings.
- Do not confuse reasoning with the final order.
- Do not repeat the instructions.
- Do not write "TASK".
- Do not write "ANSWER".
- Do not write "JUDGMENT TEXT".
- Do not include source page numbers.
- Return ONLY the requested points.
- Put EVERY point on a separate line.
- Keep every point concise but informative.

Example format:

Point 1
Point 2
Point 3
Point 4

JUDGMENT EVIDENCE:

{context}

Now provide the analysis point by point.
"""

    answer = generate_answer(
        prompt,
        max_tokens=500
    )

    points = convert_to_points(
        answer
    )

    if not points:

        points = [
            "Not clearly established from the provided judgment."
        ]

    pages = sorted(
        list(
            set(
                item["page"]
                for item in evidence
            )
        )
    )

    return {
        "points": points,
        "pages": pages
    }


# ============================================================
# ASK ABOUT CASE
# ============================================================

def ask_case_question(
    question,
    chunks,
    index
):

    evidence = retrieve_evidence(
        question,
        chunks,
        index,
        top_k=7
    )

    context = build_context(
        evidence,
        max_chars=7000
    )

    if not context:

        return (
            [
                "The judgment text provided does not clearly establish this."
            ],
            []
        )

    prompt = f"""
You are answering a question about an Indian court judgment.

QUESTION:

{question}

RULES:

- Use ONLY the supplied judgment evidence.
- Do not invent facts.
- Do not guess.
- Do not use outside information.
- Distinguish allegations from court findings.
- Answer directly.
- Do not repeat the question.
- Do not write TASK.
- Do not write ANSWER.
- Write 2 to 5 concise points.
- Put every point on a separate line.
- If unsupported, say:
  "The judgment text provided does not clearly establish this."

JUDGMENT EVIDENCE:

{context}

Provide the answer point by point.
"""

    answer = generate_answer(
        prompt,
        max_tokens=350
    )

    points = convert_to_points(
        answer
    )

    if not points:

        points = [
            "The judgment text provided does not clearly establish this."
        ]

    pages = sorted(
        list(
            set(
                item["page"]
                for item in evidence
            )
        )
    )

    return points, pages


# ============================================================
# SESSION STATE
# ============================================================

if "document_name" not in st.session_state:

    st.session_state.document_name = None

if "pages" not in st.session_state:

    st.session_state.pages = []

if "chunks" not in st.session_state:

    st.session_state.chunks = []

if "index" not in st.session_state:

    st.session_state.index = None

if "analysis" not in st.session_state:

    st.session_state.analysis = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📄 Case Document"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Judgment PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Extracting judgment..."
        ):

            pages = extract_pdf_pages(
                uploaded_file
            )

        if not pages:

            st.error(
                "No readable text was found in this PDF."
            )

            st.warning(
                "This may be a scanned PDF. "
                "OCR is required."
            )

            st.stop()

        with st.spinner(
            "Creating page-aware chunks..."
        ):

            chunks = create_chunks(
                pages
            )

        with st.spinner(
            "Building FAISS index..."
        ):

            index = create_faiss_index(
                chunks
            )

        st.session_state.document_name = (
            uploaded_file.name
        )

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.analysis = None


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.pages:

    pages = st.session_state.pages
    chunks = st.session_state.chunks

    st.markdown(
        "### 📊 Document Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            len(pages)
        )

    with col2:

        st.metric(
            "Text Chunks",
            len(chunks)
        )

    with col3:

        st.metric(
            "Retrieval",
            "FAISS + Keyword"
        )


    # ========================================================
    # CASE INTELLIGENCE
    # ========================================================

    st.markdown(
        "### ⚖️ Case Intelligence"
    )

    st.write(
        "Generate a structured evidence-grounded "
        "analysis of the uploaded judgment."
    )

    if st.button(
        "Generate Case Intelligence",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(
            0
        )

        results = {}

        sections = list(
            SECTION_CONFIG.items()
        )

        total = len(
            sections
        )

        for i, (
            section_name,
            config
        ) in enumerate(
            sections
        ):

            with st.spinner(
                f"Analyzing {section_name}..."
            ):

                results[section_name] = (
                    analyze_section(
                        section_name,
                        config,
                        chunks,
                        st.session_state.index,
                        len(pages)
                    )
                )

            progress.progress(
                (i + 1) / total
            )

        st.session_state.analysis = (
            results
        )

        st.success(
            "Case Intelligence generated successfully!"
        )


    # ========================================================
    # DISPLAY ANALYSIS
    # ========================================================

    if st.session_state.analysis:

        st.markdown(
            "## 📋 Structured Case Analysis"
        )

        for (
            section_name,
            result
        ) in st.session_state.analysis.items():

            st.markdown(
                f"### {section_name}"
            )

            points = result.get(
                "points",
                []
            )

            # ----------------------------------------------
            # DISPLAY EVERY RESULT AS A POINT
            # ----------------------------------------------

            for point in points:

                st.markdown(
                    f"- {point}"
                )

            # ----------------------------------------------
            # SOURCE PAGES ONLY
            # ----------------------------------------------

            pages_used = result.get(
                "pages",
                []
            )

            if pages_used:

                st.markdown(
                    f"""
                    <div class="source-box">
                    📚 <b>Source pages:</b>
                    {", ".join(
                        f"p. {p}"
                        for p in pages_used
                    )}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="source-box">
                    📚 <b>Source pages:</b>
                    Not clearly established
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ========================================================
    # ASK ABOUT CASE
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        "### 🔎 Ask About the Case"
    )

    question = st.text_input(
        "Ask a question about the uploaded judgment:",
        placeholder=(
            "Example: Why was the accused acquitted?"
        )
    )

    if st.button(
        "Ask Question",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the judgment..."
            ):

                answer_points, pages_used = (
                    ask_case_question(
                        question,
                        chunks,
                        st.session_state.index
                    )
                )

            # ------------------------------------------
            # ANSWER
            # ------------------------------------------

            st.markdown(
                '<div class="answer-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 💡 Answer"
            )

            for point in answer_points:

                st.markdown(
                    f"- {point}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # ------------------------------------------
            # SOURCE PAGES ONLY
            # ------------------------------------------

            if pages_used:

                st.info(
                    "📚 Source pages: "
                    + ", ".join(
                        f"p. {p}"
                        for p in pages_used
                    )
                )


else:

    st.info(
        "📄 Upload a judgment PDF from the sidebar "
        "to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    """
    <div class="footer">
    AI Case Intelligence | Evidence-Grounded Legal Judgment Analysis
    <br>
    This system provides document-based analysis and does not
    replace professional legal advice.
    </div>
    """,
    unsafe_allow_html=True
)