# 🧠 Corrective RAG

An enterprise-grade, 100% local Retrieval-Augmented Generation (RAG) pipeline optimized for CPU-bound environments. 

This project implements a sophisticated hybrid search mechanism (Dense + Sparse) and explores the architectural trade-offs between Multi-Agent Corrective RAG (CRAG) and high-speed Single-Pass RAG. It is designed to operate completely offline, ensuring absolute data privacy while maintaining high-fidelity generation.

---

## 🚀 System Architecture

This system bypasses external API dependencies by running quantized models directly on the host hardware (Ubuntu VM) using Ollama.

* **Layer 1: Inference Engine (Ollama)**
  * `llama3.1:8b` (Q4_0) for complex generation, synthesis, and instruction following.
  * `phi3.5` (3.8B) for lightweight, high-speed routing and query optimization (Phase 1).
  * `nomic-embed-text` for semantic text vectorization.
* **Layer 2: Hybrid Knowledge Store**
  * **Dense Retriever:** Dockerized Qdrant vector database for semantic meaning and conceptual matching.
  * **Sparse Retriever:** BM25 via serialized `.pkl` indices for exact keyword and serial number matching.
  * **Fusion Engine:** Reciprocal Rank Fusion (RRF) algorithm to mathematically merge and re-rank outputs, bypassing the severe latency penalties of CPU-bound Cross-Encoders.
* **Layer 3: Orchestration & Generation**
  * Custom Python backend utilizing LangChain and strict prompt engineering to force structured, hallucination-free markdown outputs.

---

## ⚖️ Architectural Trade-Offs & Engineering Evolution

Building generative AI pipelines for CPU environments requires respecting the **Memory Bandwidth Wall**. LLM inference on CPUs is rarely limited by raw compute; it is severely bottlenecked by the speed at which DDR RAM can feed model weights to the processor. 

This physical constraint drove the evolution of the project through two distinct architectural phases to achieve shipping maturity.

### Approach A: Corrective Agentic RAG (CRAG)
The initial architecture utilized a LangGraph state machine to create an autonomous, self-correcting loop.
* **The Workflow:** Hybrid Search fetches chunks -> A "Critic Node" evaluates relevance -> An "Optimizer Node" rewrites the query if context is poor -> The loop repeats until valid context is found -> Generation.
* **The Trade-off:** While theoretically highly resilient to poor user prompts, this approach resulted in unacceptable production latency (5–10 minutes). Evaluating chunks sequentially forces continuous memory reallocation. On a CPU architecture lacking high-bandwidth VRAM, the multi-step probabilistic routing overhead outweighed the retrieval benefits.

### Approach B: Fast Single-Pass Hybrid RAG (Production)
To respect user latency budgets and deliver a snappy product interface, the pipeline was pivoted to a highly optimized single-pass architecture.
* **The Workflow:** 1. Hybrid Search dynamically widens its net ($k=4$).
  2. RRF instantly fuses the results.
  3. **Context Capping:** The array is strictly sliced to the top 3 chunks to prevent context-window bloat from thrashing the CPU's memory bandwidth during the prompt evaluation phase.
  4. The context is passed directly to the generator (`llama3.1:8b`).
  5. Strict "Anti-Boilerplate" prompt engineering forces the 8B model to act as a simultaneous critic and generator, actively ignoring boilerplate and triggering a zero-hallucination guardrail if the answer is missing.
* **The Result:** By abandoning multi-step LLM routing and leveraging the speed of RRF combined with prompt-constrained generation, pipeline latency dropped from **~10 minutes to ~30-60 seconds**, while maintaining high accuracy and streaming formatted, actionable insights.

---

## 🛠️ Hardware Constraints & Future Roadmap

* **The Cross-Encoder Compromise:** In an ideal, GPU-rich environment, a Cross-Encoder (e.g., `BGE-Reranker`) would replace RRF to semantically score the exact relationship between the query and the retrieved chunks. Due to CPU inference constraints, RRF was implemented as a mathematically lightweight, hardware-aware alternative.
* **Phase 2 - Hardware Acceleration:** Future iterations will focus on PCIe Passthrough to expose a dedicated GPU to the virtualization layer, allowing the re-introduction of Cross-Encoders and asynchronous LangGraph edge routing to drive latency under 5 seconds.
* **Phase 3 - API Integration:** Wrap the production pipeline into a FastAPI backend to serve as a RESTful endpoint for frontend product interfaces.

---

## 💻 Quick Start

**1. Clone and Setup Environment**
```bash
git clone [https://github.com/your-username/Local-Multimodal-Agentic-RAG.git](https://github.com/your-username/Local-Multimodal-Agentic-RAG.git)
cd Local-Multimodal-Agentic-RAG
python3 -m venv rag-env
source rag-env/bin/activate
pip install -r requirements.txt
```
**2. Start the Vector Database**
```bash
docker compose up -d
```
**3. Ingest Documents**
Place your <code>.pdf</code> files in the <code>./docs</code> directory, then run the ingestion script to vectorize the data and build the BM25 index:
```bash
python ingest_qdrant.py
```
**4. Run the Production RAG Pipeline**
Execute the fast, single-pass hybrid search and generation:
```bash
python fast_rag.py
```
**5. (Optional) Run the Corrective Agentic RAG (CRAG) Pipeline**
If you want to test the LangGraph state machine and watch the dual-model architecture self-correct (evaluate, reject boilerplate, and rewrite queries), run the agentic version.
```bash
python agentic_rag.py
```
*Note: This approach demonstrates advanced agentic logic but incurs a significant CPU latency penalty compared to the fast single-pass pipeline.*
