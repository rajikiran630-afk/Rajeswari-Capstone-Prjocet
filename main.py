import os
from pathlib import Path
from typing import TypedDict, Literal

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# MOCK_LLM unset or 1 = required graded mock mode
# MOCK_LLM=0 = optional real LLM mode
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# =========================
# PYDANTIC MODELS
# =========================

class AskRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# =========================
# LANGGRAPH STATE
# =========================

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_documents: list[str]
    retrieved_ids: list[str]
    response: dict


# =========================
# EMBEDDINGS + CHROMADB
# =========================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    "zepto_policies"
)

print("ChromaDB collection loaded:", collection.name)


# =========================
# STRUCTURED PROMPT
# =========================

PROMPT_TEMPLATE = """
ROLE:
You are Zepto's policy support assistant.

CONTEXT:
Use only the Zepto policy context provided below.

TASK:
Answer the customer's question using the retrieved policy context.

FORMAT:
Provide a concise answer and identify the source documents used.

LENGTH:
Keep the answer short and directly relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, timings, or conditions.

FEW-SHOT EXAMPLE:

Question:
What is the delivery fee below INR 149?

Context:
Standard delivery is free on orders over INR 149;
orders below this threshold incur a flat INR 25 delivery fee.

Answer:
Orders below INR 149 incur a flat INR 25 delivery fee.

CUSTOMER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# =========================
# NODE 1: CLASSIFY INTENT
# =========================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: GraphState):

    query = state["query"].lower()

    # Required mock mode
    if MOCK_LLM:

        if any(keyword in query for keyword in POLICY_KEYWORDS):
            return {
                "intent": "policy_question"
            }

        return {
            "intent": "general_question"
        }

    # Optional real LLM mode
    return {
        "intent": "general_question"
    }


# =========================
# NODE 2: RETRIEVE + ANSWER
# =========================

def retrieve_and_answer(state: GraphState):

    query = state["query"]

    # Create query embedding
    query_embedding = model.encode(query).tolist()

    # Retrieve top 3 chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents"]
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    if not documents:

        response = AnswerResponse(
            answer="No relevant policy information was found.",
            sources=[],
            confidence=0.0
        )

        return {
            "retrieved_documents": [],
            "retrieved_ids": [],
            "response": response.model_dump()
        }

    # Required mock answer
    if MOCK_LLM:

        top_chunk_snippet = documents[0][:200]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        response = AnswerResponse(
            answer=answer,
            sources=ids,
            confidence=1.0
        )

        return {
            "retrieved_documents": documents,
            "retrieved_ids": ids,
            "response": response.model_dump()
        }

    # Optional real LLM mode
    response = AnswerResponse(
        answer="Real LLM mode is optional.",
        sources=ids,
        confidence=0.5
    )

    return {
        "retrieved_documents": documents,
        "retrieved_ids": ids,
        "response": response.model_dump()
    }


# =========================
# NODE 3: DIRECT ANSWER
# =========================

def direct_answer(state: GraphState):

    # Required mock mode
    if MOCK_LLM:

        response = AnswerResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )

        return {
            "response": response.model_dump()
        }

    # Optional real LLM mode
    response = AnswerResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )

    return {
        "response": response.model_dump()
    }


# =========================
# CONDITIONAL ROUTER
# =========================

def route_by_intent(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# =========================
# BUILD LANGGRAPH
# =========================

workflow = StateGraph(GraphState)

workflow.add_node(
    "classify_intent",
    classify_intent
)

workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

workflow.add_node(
    "direct_answer",
    direct_answer
)

workflow.add_edge(
    START,
    "classify_intent"
)

workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge(
    "retrieve_and_answer",
    END
)

workflow.add_edge(
    "direct_answer",
    END
)

graph = workflow.compile()


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0"
)


@app.get("/")
def root():

    return {
        "message": "Zepto Support Assistant is running",
        "mock_llm": MOCK_LLM
    }


@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return AnswerResponse.model_validate(
        result["response"]
    )