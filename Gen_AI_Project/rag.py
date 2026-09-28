import os

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(pdf_file):

    print("\nLoading interview questions PDF...")

    # Load PDF
    loader = PyPDFLoader(pdf_file)

    docs = loader.load()

    print(
        f"Loaded {len(docs)} pages."
    )


    # --------------------------------------------------------
    # Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        docs
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print(
        "\nCreating embeddings..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # --------------------------------------------------------
    # Store embeddings in ChromaDB
    # --------------------------------------------------------

    print(
        "Creating ChromaDB vector store..."
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print(
        "ChromaDB created successfully."
    )

    return vectordb


# ============================================================
# RAG + GROQ LLM
# ============================================================

def generate_question(vectordb):

    print(
        "\nRetrieving relevant interview questions..."
    )


    # --------------------------------------------------------
    # STEP 1: Retrieve relevant documents
    # --------------------------------------------------------

    docs = vectordb.similarity_search(
        "technical interview questions",
        k=5
    )


    if not docs:

        return "No relevant questions found."


    print(
        f"Retrieved {len(docs)} relevant documents."
    )


    # --------------------------------------------------------
    # STEP 2: Create context
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # --------------------------------------------------------
    # STEP 3: Create prompt
    # --------------------------------------------------------

    prompt = f"""
You are a Senior Technical Interviewer.

You are given interview-question information
retrieved from a knowledge base.

Use the retrieved information to generate
ONE realistic technical interview question.

Retrieved Knowledge:
--------------------
{context}
--------------------

Requirements:

1. Generate exactly ONE interview question.
2. Make the question technically relevant.
3. Do not provide the answer.
4. Do not copy the retrieved text exactly.
5. Keep the question clear and interview-ready.
6. Return only the interview question.

Interview Question:
"""


    # --------------------------------------------------------
    # STEP 4: Send context to Groq LLM
    # --------------------------------------------------------

    print(
        "\nSending retrieved context to Groq LLM..."
    )


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role": "system",

                "content": """
You are an expert technical interviewer.

Your expertise includes:

- Python
- SQL
- Machine Learning
- Deep Learning
- NLP
- Generative AI
- RAG
- LangChain

Generate professional technical interview questions.
"""
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        temperature=0.3,

        max_tokens=300
    )


    # --------------------------------------------------------
    # STEP 5: Get LLM response
    # --------------------------------------------------------

    question = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


    return question


# ============================================================
# TEST RAG + LLM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # IMPORTANT:
    # Change this path to your PDF location
    # --------------------------------------------------------

    
    pdf_file = r"C:\Users\suhit\OneDrive\Desktop\Gen_AI_Project_Datavalley\Interview Questions.pdf"

    print("=" * 70)

    print(
        "INTERVIEWGPT - RAG + LLM TEST"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Create vector database
    # --------------------------------------------------------

    vectordb = create_vector_store(
        pdf_file
    )


    # --------------------------------------------------------
    # Generate questions
    # --------------------------------------------------------

    for i in range(5):

        print("\n")
        print(
            f"INTERVIEW QUESTION {i + 1}"
        )

        print("-" * 70)


        question = generate_question(
            vectordb
        )


        print(question)

        print("-" * 70)


    print("\n")
    print(
        "RAG + LLM TEST COMPLETED"
    )