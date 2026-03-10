import time
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from hybrid_search import hybrid_search

# ==========================================
# 1. Initialize the Generator
# ==========================================
# We only need the heavy hitter now. Dropped temperature to 0.1 for strict factual extraction.
generator_llm = ChatOllama(model="phi3.5:latest", temperature=0.1)

# ==========================================
# 2. The Anti-Boilerplate Prompt
# ==========================================
PROMPT_TEMPLATE = """You are a technical knowledge assistant for ExcelLinks Corporation.
Your objective is to provide clear, fluid, and highly readable answers using ONLY the retrieved context provided below.

CRITICAL RULES:
1. TONE & STYLE: Write naturally and professionally. Do NOT sound robotic and avoid rigid structural phrases like "The technical impact is". Ensure your sentences flow smoothly.
2. FORMATTING: 
   - Start with a direct, natural introductory sentence that answers the core of the question.
   - ALWAYS use a clean markdown bulleted list to present the specific features, capabilities, or components found in the context. Keep the bullets concise.
   - Conclude with a brief, natural thought that seamlessly transitions into a relevant follow-up question (e.g., "Would you like me to elaborate on the [Specific Feature]?").
3. ZERO HALLUCINATION GUARDRAIL: You must not infer, guess, or synthesize information about subjects not explicitly detailed in the context. If the provided context does not explicitly contain the answer, you MUST halt and output EXACTLY the following string and nothing else:
"I do not have enough information in the current documentation to answer that fully. Could you provide more detail or ask about another specific module?"

Question: {question} \n
Context: {context} \n
Answer:"""

def fast_rag_pipeline(query: str, k_docs: int = 4):
    """Executes a single-pass hybrid retrieval and generation."""
    start_time = time.time()
    
    # 1. Fetch top 4 from each, fuse them, but STRICTLY keep only the top 3 overall.
    print(f"\n[1/2] Executing Hybrid Retrieval for: '{query}'...")
    
    # THE FIX: Slice the fused array so we don't nuke the CPU's memory bandwidth
    docs = hybrid_search(query, k=k_docs)[:3] 
    
    if not docs:
        return "I do not have enough information in the current documentation to answer that fully. Could you provide more detail or ask about another specific module?"
        
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # 2. Single-Pass Generation
    print(f"[2/2] Synthesizing Answer with phi3.5:latest...")
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
