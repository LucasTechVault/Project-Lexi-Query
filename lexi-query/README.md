# Project LexiQuery - Financial RAG Pipeline

## Project Overview

LexiQuery is a evaluation-driven RAG system designed to answer complex financial questions using dense corporate filings (10-Ks and 10-Qs).

Rather than relying on naive "chat-with-PDF" wrappers, this project was built from scratch to focus on RAGOps, modular architecture and systematic evaluation. It features a distributed architecture, running a lightweight client locally while offloading heavy LLM inference to a remote multi-GPU (dual H100 NVL) cluster.

## Architecture & Infrastructure

The system operates on a split-architecture model to optimize for both development speed and compute power.

- **Local App (Macbook)**: Handles data processing, chunking, embedding generation (BAAI/bge-large-en-v1.5), ChromaDB vector storage, and orchestration scripts.

- **Remote Inference Server (2x Nvidia H100 NVL)**: Hosts the generative LLMs (Llama-3-70B / Qwen 32B) using vLLM for high-throughput, low-latency text generation.

- **Network Bridge**: Local app communicates with remote vLLM instance via OpenAI-compatible API layer routed over Cisco VPN, with explicit host-binding (0.0.0.0).

## Implementation Phases

### Phase 1: Data Handling & Processing

- **Dataset:** FinanceBench dataset for retrieval evaluation and sourced corresponding raw 10-K / 10-Q pdfs from document links provided in dataset.

- **Parsing & Chunking:** Implemented ingestion pipeline that parses financial PDFs page-by-page. Applied overlapping chunking strategy (800-words chunk, 120-word overlap) to preserve financial context across page breaks.

- **Embedding:** Utilized encoder-only BGE-large model to convert chunks into 1024-dim semnatic vectors

### Phase 2: Search Engineering (Retriever)

- Built custom `DenseRetriever` class wrapping ChromaDB
- Established 4 pillars of vector storage, ensuring metadata (doc_id, page_num) is bounded to vector for downstream validation

### Phase 3: Generation Engineering (Orchestrator)

- **Grounded Prompting:** Designed strict prompt contract enforcing 2 critical rules:

  1. Abstention: Model must refuse to answer if conteext is insufficient (prevent hallucinations)
  2. Citation: Model must cite exact [doc_id p.X] in metadata to formulate answers.

- **Orchestrator:** Built `RagAnswerer` to pipe retrieved chunks into vLLM context window, returning a payload containing both synthesized answer and verifiable source chunks.

### Phase 4: Evaluation Engineering (RAGOps)

Before grading the LLM's generated answer, the retrieval system was rigorously tested in isolation to establish baseline success ceiling. We cannot evaluate LLM generation ability if we feed bad context knowledge to it.

| Metric                     | Score  | Insight                                                                                  |
| -------------------------- | ------ | ---------------------------------------------------------------------------------------- |
| Doc Hit@5 Rate             | 78.7%  | Dense retriever successfully places correct document in top-5 chunks nearly 80% of time. |
| Mean Reciprocal Rank (MRR) | 0.5661 | When correct chunk is found, system highly prioritize it, typically rank 1 or rank 2.    |

## Key Engineering Findings & Challenges

### 1. Cross-Contamination Problem (Semantic vs Keyword)

- Observation: During evaluation, dense retriever struggled to differentiate between companies when asked specific financial questions (pulling Nike's earnings when asked for Apple's)
- Conclusion: Pure dense semantic search over-indexes on concepts and under-indexes on specific nouns

### 2. Distributed Network Hurdles

-
