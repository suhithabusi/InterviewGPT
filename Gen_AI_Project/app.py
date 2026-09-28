# app.py
import streamlit as st
import tempfile
import os

from resume_parser import extract_resume_text
from skill_extractor import extract_skills

from rag import create_vector_store, generate_question

from evaluator import evaluate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InterviewGPT",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 InterviewGPT")

st.subheader(
    "AI Interview Coach using RAG + Groq LLM"
)

st.markdown("---")


# ============================================================
# SESSION STATE
# ============================================================

if "question" not in st.session_state:
    st.session_state.question = None

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "skills" not in st.session_state:
    st.session_state.skills = []


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Upload Resume")

resume = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"],
    key="resume_uploader"
)


if resume is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(
            resume.read()
        )

        resume_path = tmp.name


    # Extract resume text

    resume_text = extract_resume_text(
        resume_path
    )

    st.session_state.resume_text = resume_text


    st.success(
        "Resume uploaded successfully!"
    )


    # --------------------------------------------------------
    # Resume Preview
    # --------------------------------------------------------

    st.subheader("📋 Resume Preview")

    st.text(
        resume_text[:1500]
    )


    # --------------------------------------------------------
    # Skill Extraction
    # --------------------------------------------------------

    skills = extract_skills(
        resume_text
    )

    st.session_state.skills = skills


    st.subheader("🧠 Detected Skills")


    if skills:

        st.success(
            ", ".join(skills)
        )

    else:

        st.warning(
            "No technical skills detected."
        )


st.markdown("---")


# ============================================================
# INTERVIEW QUESTION PDF
# ============================================================

st.header("📚 Interview Question Knowledge Base")


interview_pdf = st.file_uploader(
    "Upload Interview Questions PDF",
    type=["pdf"],
    key="interview_pdf_uploader"
)


if interview_pdf is not None:

    st.success(
        "Interview question PDF uploaded!"
    )


    if st.button(
        "🔎 Build RAG Knowledge Base"
    ):

        with st.spinner(
            "Creating ChromaDB vector database..."
        ):

            # Save uploaded PDF temporarily

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(
                    interview_pdf.read()
                )

                pdf_path = tmp.name


            # Create vector database

            vectordb = create_vector_store(
                pdf_path
            )


            # Store vector DB in session

            st.session_state.vectordb = vectordb


        st.success(
            "RAG knowledge base created successfully!"
        )


# ============================================================
# GENERATE INTERVIEW QUESTION
# ============================================================

st.markdown("---")

st.header("🎯 Generate Interview Question")


if st.session_state.vectordb is None:

    st.info(
        "Please upload the Interview Questions PDF "
        "and build the RAG knowledge base first."
    )

else:

    if st.button(
        "🤖 Generate Interview Question"
    ):

        with st.spinner(
            "Retrieving knowledge and generating question..."
        ):

            question = generate_question(
                st.session_state.vectordb
            )


            st.session_state.question = question


        st.success(
            "Interview question generated!"
        )


# ============================================================
# DISPLAY QUESTION
# ============================================================

if st.session_state.question:

    st.markdown("---")

    st.header(
        "🎤 Interview Question"
    )


    st.info(
        st.session_state.question
    )


    # ========================================================
    # CANDIDATE ANSWER
    # ========================================================

    st.subheader(
        "✍️ Your Answer"
    )


    answer = st.text_area(
        "Type your interview answer below:",
        height=220,
        placeholder=(
            "Explain your answer as if you are "
            "speaking to a technical interviewer..."
        )
    )


    # ========================================================
    # EVALUATE ANSWER
    # ========================================================

    if st.button(
        "📊 Evaluate My Answer"
    ):

        if not answer.strip():

            st.warning(
                "Please enter your answer first."
            )

        else:

            with st.spinner(
                "Senior AI Interviewer is evaluating your answer..."
            ):

                feedback = evaluate_answer(
                    st.session_state.question,
                    answer
                )


            st.markdown("---")

            st.header(
                "📊 Interview Evaluation"
            )


            st.markdown(
                feedback
            )


# ============================================================
# GENERATE ANOTHER QUESTION
# ============================================================

if st.session_state.vectordb is not None:

    st.markdown("---")


    if st.button(
        "🔄 Generate Another Question"
    ):

        with st.spinner(
            "Generating another question..."
        ):

            question = generate_question(
                st.session_state.vectordb
            )


            st.session_state.question = question


        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "InterviewGPT | Resume Analysis | "
    "RAG | ChromaDB | Groq LLM"
)