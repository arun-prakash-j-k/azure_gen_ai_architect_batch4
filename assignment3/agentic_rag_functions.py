# agentic_rag_azure.py

import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
import mlflow
from pinecone import Pinecone
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# Load environment variables and initialize tools
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# --- 0. State, Models, and Pydantic Schemas ---

class Snippet(TypedDict):
    """Structure for a retrieved KB snippet."""
    id: str
    text: str

class AgentState(TypedDict):
    """State for the LangGraph workflow."""
    query: str
    snippets: List[Snippet]
    initial_answer: str
    critique_result: str
    final_answer: str

# Pydantic Schema for Critique Node Output (forces structured output)
class Critique(BaseModel):
    """Critique model for the initial answer."""
    binary_score: str = Field(description="Must be 'COMPLETE' if the answer is thorough and addresses the query using the snippets, or 'REFINE' if it is lacking and needs more context.")
    reasoning: str = Field(description="Brief explanation for the score.")

# --- 1. LLM and Vector DB Setup ---

# Azure GPT-4 mini for Generation (Temperature=0)
azure_llm = AzureChatOpenAI(
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0
)

# Gemini for Self-Critique (using LangChain wrapper for structured output)
# Note: Using Gemini as requested in the assignment for the critique step.
gemini_critique_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY")
).with_structured_output(Critique)

# Azure Embeddings for Retrieval
embeddings_model = AzureOpenAIEmbeddings(
    model = os.getenv("EMBEDDING_MODEL_NAME"),
    deployment = os.getenv("EMBEDDING_DEPLOYMENT"), 
    azure_endpoint = os.getenv("EMBEDDING_ENDPOINT"),
    api_key = os.getenv("EMBEDDING_API_KEY"), 
    api_version = os.getenv("EMBEDDING_API_VERSION")
)

# Pinecone Client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# --- 2. LangGraph Node Functions ---

# 2.1 Retriever Node
def retrieve_snippets(state: AgentState, k: int) -> List[Snippet]:
    """Retrieves top-k snippets from Pinecone."""
    query_vector = embeddings_model.embed_query(state["query"])
    
    # Query Pinecone
    results = pinecone_index.query(
        vector=query_vector, 
        top_k=k, 
        include_metadata=True
    )
    
    snippets = [
        Snippet(
            id=match.metadata.get("id", "KBXXX"),
            text=match.metadata.get("page_content", "No content found")
        )
        for match in results.matches
    ]
    return snippets

def retriever_node(state: AgentState) -> AgentState:
    """Node 1: Retrieves top 5 snippets and updates state."""
    print("--- RETRIEVER NODE: Fetching top 5 snippets ---")
    snippets = retrieve_snippets(state, k=5)
    
    # Log retrieved snippets to MLflow
    mlflow.log_dict({"initial_snippets": [s['id'] for s in snippets]}, "artifacts/retrieval_logs.json")
    
    return {"snippets": snippets}

# 2.2 LLM Answer Node
ANSWER_PROMPT = """
You are an expert technical writer. Use the provided Knowledge Base (KB) snippets to answer the user's question.
Your answer must be professional, accurate, and directly cite every fact using the [KBxxx] format.
If a concept is mentioned in multiple snippets, cite all of them. Do not use any external knowledge.

QUESTION: {question}

KNOWLEDGE BASE SNIPPETS:
{context}

ANSWER:
"""

def llm_answer_node(state: AgentState) -> AgentState:
    """Node 2: Generates the initial answer."""
    print("--- LLM ANSWER NODE: Generating initial answer ---")
    
    context_str = "\n---\n".join([f"[{s['id']}] {s['text']}" for s in state["snippets"]])
    
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT).format(
        question=state["query"],
        context=context_str
    )
    
    response = azure_llm.invoke(prompt)
    initial_answer = response.content
    
    # Log initial answer to MLflow
    mlflow.log_text(initial_answer, "artifacts/initial_answer.txt")
    
    return {"initial_answer": initial_answer}

# 2.3 Self-Critique Node
CRITIQUE_PROMPT = """
You are a RAG system critique agent. Your task is to evaluate an AI-generated answer based on a user's question and the context provided.
The critique must be a binary score: 'COMPLETE' or 'REFINE'.

Criteria for 'COMPLETE':
1. The answer thoroughly addresses all parts of the user's question.
2. The answer is grounded *only* in the provided KB snippets.
3. Every factual claim is correctly cited with the [KBxxx] format.

Criteria for 'REFINE':
1. The answer is too brief or misses key information relative to the user's question.
2. The answer could be significantly improved by incorporating more detail, which is likely available in the vector database.
3. The answer is well-written but the question is complex, suggesting more context might be needed.

User Question: {question}
KB Snippets Used (IDs): {snippet_ids}
AI Answer: {answer}

Critique the answer and output the Pydantic schema strictly.
"""

def self_critique_node(state: AgentState) -> AgentState:
    """Node 3: Critiques the initial answer."""
    print("--- SELF-CRITIQUE NODE: Evaluating answer ---")
    
    snippet_ids = [s['id'] for s in state["snippets"]]
    
    prompt = ChatPromptTemplate.from_template(CRITIQUE_PROMPT).format(
        question=state["query"],
        snippet_ids=", ".join(snippet_ids),
        answer=state["initial_answer"]
    )
    
    critique_output: Critique = gemini_critique_llm.invoke(prompt)
    critique_result = critique_output.binary_score
    
    # Log critique result to MLflow
    mlflow.log_param("critique_decision", critique_result)
    mlflow.log_text(critique_output.reasoning, "artifacts/critique_reasoning.txt")
    
    return {"critique_result": critique_result}

# 2.4 Refinement Node
REFINEMENT_PROMPT = """
You are an expert technical writer. You need to **REFINE** the initial answer using **ALL** of the provided Knowledge Base (KB) snippets, including the new one.
Your refined answer must be more comprehensive than the initial one.
Ensure your answer is professional, accurate, and directly cites every fact using the [KBxxx] format.

User Question: {question}
ALL KB SNIPPETS:
{context}

INITIAL ANSWER TO REFINE:
{initial_answer}

REFINED ANSWER:
"""

def refinement_node(state: AgentState) -> AgentState:
    """Node 4: Retrieves 1 more snippet and regenerates the answer."""
    print("--- REFINEMENT NODE: Retrieving 1 more snippet & refining ---")
    
    # Retrieve the 6th snippet (k=6, but we only need the addition)
    all_6_snippets = retrieve_snippets(state, k=6)
    new_snippet = all_6_snippets[-1]
    
    # Append the new snippet to the state (if it's new)
    if new_snippet['id'] not in [s['id'] for s in state['snippets']]:
        state["snippets"].append(new_snippet)
        mlflow.log_dict({"additional_snippet": new_snippet['id']}, "artifacts/retrieval_logs.json")
        print(f"Retrieved and added new snippet: {new_snippet['id']}")
    else:
        print("Warning: Top 6 snippets were the same as top 5. Refining with existing context.")
    
    context_str = "\n---\n".join([f"[{s['id']}] {s['text']}" for s in state["snippets"]])
    
    prompt = ChatPromptTemplate.from_template(REFINEMENT_PROMPT).format(
        question=state["query"],
        context=context_str,
        initial_answer=state["initial_answer"]
    )
    
    response = azure_llm.invoke(prompt)
    refined_answer = response.content
    
    # Log refined answer to MLflow
    mlflow.log_text(refined_answer, "artifacts/refined_answer.txt")
    
    return {"final_answer": refined_answer}

# --- 3. Conditional Edge Logic ---

def route_critique(state: AgentState) -> str:
    """Conditional router based on the critique result."""
    print(f"--- ROUTER: Critique result is: {state['critique_result']} ---")
    if state["critique_result"] == "REFINE":
        return "refine"
    return "complete"

# --- 4. Build and Compile LangGraph ---

def build_graph():
    """Defines and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 4.1 Define Nodes
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("answer", llm_answer_node)
    workflow.add_node("critique", self_critique_node)
    workflow.add_node("refine", refinement_node)

    # 4.2 Define Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", "critique")

    # Conditional Edge (Decision Logic)
    workflow.add_conditional_edges(
        "critique",
        route_critique,
        {
            "refine": "refine",
            "complete": END
        }
    )

    # Final Path
    workflow.add_edge("refine", END)
    
    return workflow.compile()

# --- 5. Main Execution and MLflow Tracking ---

def run_agentic_rag(compiled_graph, query: str):
    """Executes the graph for a query, logging with MLflow."""
    print(f"\n--- Starting RAG for Query: '{query}' ---")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Query: {query[:30]}...") as run:
        mlflow.log_param("input_query", query)
        
        # Execute the graph
        initial_state = {"query": query, "snippets": [], "initial_answer": "", "critique_result": "", "final_answer": ""}
        final_state = compiled_graph.invoke(initial_state)
        
        # Determine the final answer
        if final_state.get("final_answer"):
            final_response = final_state["final_answer"]
            mlflow.log_param("final_step", "Refinement")
        else:
            final_response = final_state["initial_answer"]
            mlflow.log_param("final_step", "Initial Answer (Complete)")

        # Log the final output
        mlflow.log_text(final_response, "artifacts/final_response.txt")
        print("\n--- FINAL RESPONSE ---")
        print(final_response)
        
    return final_response