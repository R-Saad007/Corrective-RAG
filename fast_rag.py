import time
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from hybrid_search import hybrid_search

# ==========================================
# 1. Initialize the Generator
# ==========================================
# We only need the heavy hitter now. Dropped temperature to 0.1 for strict factual extraction.
generator_llm = ChatOllama(
    model="qwen2.5:0.5b", 
    temperature=0.1,
    # THE FIX: Hard-cap the context window so it never bloats memory
    num_ctx=2048 
)

# ==========================================
# 2. The Anti-Boilerplate Prompt
# ==========================================
PROMPT_TEMPLATE = """You are AxIn Help, the expert technical knowledge assistant for AxIn.
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

Answer:"""

def fast_rag_pipeline(query: str, k_docs: int = 4):
    """Executes a single-pass hybrid retrieval and generation."""
    start_time = time.time()
    
    # 1. Fetch top 4 from each, fuse them, but STRICTLY keep only the top 3 overall.
    print(f"\n[1/2] Executing Hybrid Retrieval for: '{query}'...")
    
    # THE FIX: Slice the fused array so we don't nuke the CPU's memory bandwidth
    docs = hybrid_search(query, k=k_docs)[:2] 
    
    if not docs:
        return "I do not have enough information in the current documentation to answer that fully. Could you provide more detail or ask about another specific module?"
        
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # 2. Single-Pass Generation
    print(f"[2/2] Synthesizing Answer with qwen2.5:0.5b...")
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
    chain = prompt | generator_llm | StrOutputParser()
    
    print("\n==========================================")
    print("GENERATING RESPONSE:")
    print("==========================================\n")
    
    full_response = ""
    for chunk in chain.stream({"question": query, "context": context}):
        print(chunk, end="", flush=True)
        full_response += chunk
        
    end_time = time.time()
    print(f"\n\n==========================================")
    print(f"Total Pipeline Latency: {end_time - start_time:.2f} seconds")
    print("==========================================")
    
    return full_response

if __name__ == "__main__":
    # Test the pipeline with the exact query that caused the loop
    test_query = "What does the ClickOps module do?"
    fast_rag_pipeline(test_query)
