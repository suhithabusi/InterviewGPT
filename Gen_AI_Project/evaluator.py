import os

from dotenv import load_dotenv
from groq import Groq


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
# ANSWER EVALUATION
# ============================================================

def evaluate_answer(question, answer):

    prompt = f"""
You are a Senior Technical Interviewer.

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluate the candidate's answer.

Provide the following:

1. Technical Score (out of 10)
2. Communication Score (out of 10)
3. Strengths
4. Weaknesses
5. Missing Concepts
6. Improved Answer
7. Final Recommendation

Be specific and constructive.

Do not give a score without explaining the reasoning.
"""


    # ========================================================
    # CALL GROQ LLM
    # ========================================================

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role": "system",

                "content": """
You are an expert senior technical interviewer.

You evaluate candidates in:

- Python
- SQL
- Machine Learning
- Deep Learning
- NLP
- Generative AI
- RAG
- LangChain
- Transformers

Give professional and constructive interview feedback.
"""
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        temperature=0.2,

        max_tokens=1500
    )


    # ========================================================
    # RETURN FEEDBACK
    # ========================================================

    feedback = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return feedback


# ============================================================
# TEST EVALUATOR
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INTERVIEW ANSWER EVALUATOR")
    print("=" * 70)


    question = input(
        "\nEnter Interview Question:\n\n"
    )


    answer = input(
        "\nEnter Candidate Answer:\n\n"
    )


    result = evaluate_answer(
        question,
        answer
    )


    print("\n")
    print("=" * 70)
    print("INTERVIEW EVALUATION REPORT")
    print("=" * 70)

    print(result)