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
    temperature=0.2,
    # THE FIX: Hard-cap the context window so it never bloats memory
    num_ctx=2048,
    repeat_penalty=1.2 
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
- Start directly with a brief, definitive explanatory paragraph. ABSOLUTELY NO introductory filler phrases. Jump straight into the facts.
- FULL NAVIGATION PATH REQUIRED: If the action occurs inside a sub-module, tool, or specific tab (such as the "Playground" inside "Site Specific View"), you MUST explicitly state the exact location and path the user needs to take.
- Structure the core of your answer using a clean markdown bulleted list. Break down complex information step-by-step.
- Follow the bullet points with a brief concluding sentence summarizing the value of this information.
- NEVER include markdown image syntax like ![](url), HTML tags, or placeholder URLs in your response.
- DO NOT include reasoning, meta-explanation, or internal thinking. 
- ALWAYS end with a highly relevant follow-up question to keep the conversation going. Do NOT use the word "could" in your follow-up question.

Answer:"""

def fast_rag_pipeline(query: str, k_docs: int = 4):
    """Executes a single-pass hybrid retrieval and generation."""
    start_time = time.time()
    
    # 1. Fetch top 4 from each, fuse them, but STRICTLY keep only the top 3 overall.
    print(f"\n[1/2] Executing Hybrid Retrieval for: '{query}'...")
    
    # THE FIX: Slice the fused array so we don't nuke the CPU's memory bandwidth
    docs = hybrid_search(query, k=k_docs)[:4] 
    
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

def stream_rag_pipeline(query_text: str):
    """
    Executes the RAG pipeline and yields the answer token-by-token.
    """
    print(f"\n[1/2] Executing Hybrid Retrieval for: '{query_text}'...")
    # 1. Retrieve the documents
    docs = hybrid_search(query_text, k=4)[:4]
    
    if not docs:
        yield "I do not have enough information in the current documentation to answer that fully. Please provide more detail or ask about another specific module."
        return
        
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    print("[2/2] Streaming Answer from qwen2.5:0.5b...")
    
    # 2. Build the LangChain Prompt object (THE FIX)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
    
    # 3. Build the chain
    chain = prompt | generator_llm 
    
    # 4. Stream the output token by token
    for chunk in chain.stream({"context": context_text, "question": query_text}):
        # chunk.content contains the raw string of the generated token
        yield chunk.content


if __name__ == "__main__":
    # Test the pipeline with the exact query that caused the loop
    test_query = "What does the ClickOps module do?"
    fast_rag_pipeline(test_query)
    test_query = "How can I view detailed information for sites affected by Critical alarms in e-NOC?"
    fast_rag_pipeline(test_query)
    test_query = "What file format can I download the Power Consumption report in from the Reporting Hub?"
    fast_rag_pipeline(test_query)
    test_query = "What specific metrics does the Battery Performance category evaluate in ClickOPS?"
    fast_rag_pipeline(test_query)
    test_query = "How do I save a custom query for future use in the Site Specific View?"
    fast_rag_pipeline(test_query)
    test_query = "What happens when I click 'Go to Site Snapshot' inside Site Watch?"
    fast_rag_pipeline(test_query)