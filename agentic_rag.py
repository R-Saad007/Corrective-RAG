from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from hybrid_search import hybrid_search

# ==========================================
# 1. Define the Graph State
# ==========================================
class GraphState(TypedDict):
    """
    Represents the state of our agentic loop.
    """
    question: str
    generation: str
    documents: List[str]
    loop_count: int

# ==========================================
# 2. Initialize the Dual-Brain LLMs
# ==========================================
# The "Fast Thinker" for routing and grading
evaluator_llm = ChatOllama(model="phi3.5:latest", temperature=0)

# The "Heavy Generator" for final synthesis
generator_llm = ChatOllama(model="llama3.1:8b", temperature=0.3)

# ==========================================
# 3. Define the Nodes (The Actions)
# ==========================================
def retrieve_node(state: GraphState):
    print("\n--- NODE: RETRIEVE ---")
    question = state["question"]
    loop_count = state.get("loop_count", 0)
    
    # Call the Layer 2 hybrid search we built
    docs = hybrid_search(question, k=3)
    # Extract just the text content to keep the state lightweight
    doc_texts = [doc.page_content for doc in docs]
    
    return {"documents": doc_texts, "question": question, "loop_count": loop_count}

def grade_documents_node(state: GraphState):
    print("\n--- NODE: GRADE DOCUMENTS (Critic) ---")
    question = state["question"]
    
    # CRITICAL CPU FIX: Only grade the top 2 chunks from the hybrid search.
    # This cuts our LLM evaluation time by 66%.
    documents = state["documents"][:2] 
    loop_count = state["loop_count"]
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
        Provide ONLY the word 'yes' or 'no'. DO NOT provide any other text.""",
        input_variables=["question", "document"],
    )
    
    chain = prompt | evaluator_llm | StrOutputParser()
    
    relevant_docs = []
    # This loop will now only run a maximum of 2 times
    for doc in documents:
        score = chain.invoke({"question": question, "document": doc}).strip().lower()
        if "yes" in score:
            print("   - Grader: Document is RELEVANT.")
            relevant_docs.append(doc)
        else:
            print("   - Grader: Document is IRRELEVANT (Boilerplate detected).")
            
    return {"documents": relevant_docs, "question": question, "loop_count": loop_count}

def rewrite_query_node(state: GraphState):
    print("\n--- NODE: REWRITE QUERY ---")
    question = state["question"]
    loop_count = state["loop_count"] + 1
    
    prompt = PromptTemplate(
        template="""You are an expert query optimizer for an enterprise search system. 
        Your task is to take a vague or short user question and rewrite it into a highly specific, clear, and comprehensive search query. 
        Focus strictly on expanding the core subject of the question (e.g., adding implicit context like "function", "purpose", "components", or "architecture").
        
        CRITICAL RULES:
        1. Do NOT add meta-phrases like "in a vector database", "search query", or "database retrieval" to your output.
        2. Do NOT answer the question. Only rewrite it.
        3. Provide ONLY the rewritten question. DO NOT provide any other text, quotes, or explanation.
        
        Initial question: {question} \n
        Rewritten question:""",
        input_variables=["question"],
    )
    
    chain = prompt | evaluator_llm | StrOutputParser()
    better_question = chain.invoke({"question": question}).strip()
    
    print(f"   - Original: {question}")
    print(f"   - Rewritten: {better_question}")
    
    return {"documents": state["documents"], "question": better_question, "loop_count": loop_count}

def generate_node(state: GraphState):
    print("\n--- NODE: GENERATE FINAL ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(documents)
    
    prompt = PromptTemplate(
        template="""You are AxIn Help, the expert technical knowledge assistant for AxIn.
        Your ONLY source of knowledge is the provided Context. 

        <context>
        {context}
        </context>

        Question: {question}

        STRICT EVALUATION PROTOCOL:
        First, determine if the provided Context explicitly contains the answer to the Question.

        IF THE ANSWER IS NOT IN THE CONTEXT:
        You are forbidden from guessing, inferring, or using outside knowledge. You MUST output EXACTLY the following string and nothing else. Stop generating immediately after this string:
        "I do not have enough information in the current documentation to answer that fully. Please provide more detail or ask about another specific module."

        IF THE ANSWER IS IN THE CONTEXT:
        Format your response exactly following these structural rules. DO NOT number your paragraphs.
        - Start with a brief, friendly explanatory paragraph setting the context for your answer. Do NOT start this paragraph with a number. Do NOT use speculative words like "seems", "appears", or "likely". Be definitive and factual.
        - Structure the core of your answer using a clean markdown bulleted list. Break down complex information step-by-step.
        - Follow the bullet points with a brief concluding sentence summarizing the value of this information.
        - ALWAYS end with a highly relevant follow-up question to keep the conversation going. Do NOT use the word "could" in your follow-up question (e.g., use "Would you like me to explain...", "Do you need...").

        Answer:""",
        input_variables=["question", "context"],
    )
    
    chain = prompt | generator_llm | StrOutputParser()
    generation = chain.invoke({"question": question, "context": context})
    
    return {"documents": documents, "question": question, "generation": generation, "loop_count": state["loop_count"]}

# ==========================================
# 4. Define the Conditional Edges (The Logic)
# ==========================================
def decide_to_generate(state: GraphState):
    print("\n--- LOGIC: DECIDE TO GENERATE OR REWRITE ---")
    relevant_docs = state["documents"]
    loop_count = state["loop_count"]
    
    # If the grader found at least one relevant document, proceed to generation
    if relevant_docs:
        print("   - Decision: Proceeding to Generation.")
        return "generate"
    # If all docs were boilerplate/irrelevant, and we haven't looped too many times, rewrite
    elif loop_count < 2:
        print("   - Decision: Context is garbage. Rewriting Query.")
        return "rewrite_query"
    # Fallback to prevent infinite loops
    else:
        print("   - Decision: Max loops reached. Forcing Generation.")
        return "generate"

# ==========================================
# 5. Compile the Graph
# ==========================================
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

# Build the graph edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    }
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

# Compile into a runnable application
app = workflow.compile()

if __name__ == "__main__":
    # Test the Agent with the tricky query
    tricky_query = "What does ClickOps do?"
    
    # Initialize the state
    initial_state = {"question": tricky_query, "loop_count": 0}
    
    print(f"\n[SYSTEM START] Executing Agentic Loop for Query: '{tricky_query}'")
    
    # Stream the events so we can watch the agent think
    for output in app.stream(initial_state):
        for key, value in output.items():
            pass # The print statements inside the nodes handle the logging
            
    print("\n==========================================")
    print("FINAL GENERATED RESPONSE:")
    print("==========================================")
    print(value["generation"])
