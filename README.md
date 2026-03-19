# Enterprise RAG Platform on Azure

**Production-grade Retrieval-Augmented Generation architecture for enterprise knowledge management**

> *Reference architecture + working prototype demonstrating how to build a governed, scalable RAG system on Azure — the pattern most enterprises need but few implement correctly.*

---

## What This Is

This project implements a full enterprise RAG (Retrieval-Augmented Generation) platform on Azure — the architecture pattern that sits behind most real-world generative AI deployments in large organizations. It's not a chatbot demo. It's the infrastructure layer: document ingestion pipelines, vector indexing, retrieval logic, LLM orchestration, and the governance controls that make it enterprise-safe.

**The problem it solves:** Most RAG tutorials show you how to chat with a PDF. Enterprise RAG requires chunking strategy decisions, multi-source retrieval, access control enforcement, citation tracing, hallucination guardrails, and cost controls — none of which appear in the tutorials. This reference architecture handles all of them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│  Documents (PDF/Word/SharePoint)                                │
│       │                                                         │
│       ▼                                                         │
│  Azure Document Intelligence ──► Chunking Engine               │
│       │                              │                          │
│       │                    (semantic chunking,                  │
│       │                     overlap control,                    │
│       │                     metadata extraction)                │
│       ▼                              ▼                          │
│  Azure Blob Storage ◄────── Chunk Store (with metadata)         │
│                                      │                          │
│                                      ▼                          │
│                           Azure AI Search (vector index)        │
└─────────────────────────────────────────────────────────────────┘
                                       │
                                       │ retrieval
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                              │
│                                                                 │
│  User Query                                                     │
│       │                                                         │
│       ▼                                                         │
│  Query Rewriting ──► Embedding (text-embedding-3-large)         │
│                              │                                  │
│                              ▼                                  │
│              Hybrid Search (vector + keyword)                   │
│                              │                                  │
│                    Top-K chunks retrieved                       │
│                              │                                  │
│                              ▼                                  │
│            Reranker (cross-encoder scoring)                     │
│                              │                                  │
│                              ▼                                  │
│        Prompt Assembly ──► Azure OpenAI GPT-4o                  │
│                              │                                  │
│                              ▼                                  │
│         Response + Citations + Confidence Score                 │
└─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GOVERNANCE LAYER                            │
│                                                                 │
│  • Access control enforcement (Azure AD RBAC)                   │
│  • PII detection before ingestion                               │
│  • Citation tracing (every response → source chunks)           │
│  • Hallucination scoring (groundedness check)                   │
│  • Cost tracking (tokens per query, per user, per department)  │
│  • Audit log (Azure Monitor + Log Analytics)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Chunking strategy
Semantic chunking over fixed-size chunking — respects sentence and paragraph boundaries, which significantly improves retrieval precision for enterprise documents (policies, contracts, technical specs).

### Hybrid search
Vector similarity alone misses exact-match queries (product codes, policy numbers, names). Hybrid search (vector + BM25 keyword) with RRF (Reciprocal Rank Fusion) merging outperforms pure vector search for enterprise content by 15-20% on precision@5.

### Reranking
Cross-encoder reranking as a second-pass filter is the single highest-ROI improvement in a RAG pipeline. The bi-encoder retrieval is fast but imprecise; the cross-encoder reranker is slow but accurate — running it only on top-20 candidates keeps latency acceptable.

### Governance controls
Enterprise RAG without access control is a data leak waiting to happen. This implementation enforces document-level permissions at retrieval time — a user can only receive chunks from documents they have Azure AD access to, regardless of what they ask.

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Azure OpenAI (GPT-4o) |
| Embeddings | Azure OpenAI (text-embedding-3-large) |
| Vector Store | Azure AI Search (with vector index) |
| Orchestration | LangChain + Python |
| Document Processing | Azure Document Intelligence |
| Storage | Azure Blob Storage |
| Access Control | Azure Active Directory / Entra ID |
| Observability | Azure Monitor + Application Insights |
| Infrastructure | Azure Bicep (IaC) |

---

## Project Structure

```
01-enterprise-rag-platform/
├── README.md
├── architecture/
│   ├── architecture-decision-records/
│   │   ├── ADR-001-chunking-strategy.md
│   │   ├── ADR-002-hybrid-search.md
│   │   └── ADR-003-access-control-approach.md
│   └── diagrams/
├── src/
│   ├── ingestion/
│   │   ├── document_processor.py       # Document Intelligence wrapper
│   │   ├── chunker.py                  # Semantic chunking engine
│   │   ├── embedder.py                 # Embedding generation
│   │   └── indexer.py                  # Azure AI Search indexing
│   ├── retrieval/
│   │   ├── query_rewriter.py           # Query expansion/rewriting
│   │   ├── hybrid_retriever.py         # Vector + keyword retrieval
│   │   ├── reranker.py                 # Cross-encoder reranking
│   │   └── access_enforcer.py          # RBAC enforcement at retrieval
│   ├── generation/
│   │   ├── prompt_builder.py           # Prompt assembly with context
│   │   ├── llm_client.py               # Azure OpenAI wrapper
│   │   └── citation_tracer.py          # Source attribution
│   ├── governance/
│   │   ├── pii_detector.py             # PII scanning pre-ingestion
│   │   ├── groundedness_checker.py     # Hallucination detection
│   │   └── cost_tracker.py             # Token usage by user/dept
│   └── api/
│       └── app.py                      # FastAPI endpoint
├── infra/
│   └── main.bicep                      # Azure infrastructure as code
├── notebooks/
│   ├── 01-ingestion-pipeline-demo.ipynb
│   ├── 02-retrieval-evaluation.ipynb
│   └── 03-rag-evaluation-ragas.ipynb
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_groundedness.py
├── evaluation/
│   └── rag_eval_dataset.json           # Evaluation Q&A pairs
└── requirements.txt
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/johndoe/enterprise-rag-platform

# Install dependencies
pip install -r requirements.txt

# Configure Azure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI endpoint, key, Search endpoint

# Run the ingestion pipeline on sample documents
python src/ingestion/document_processor.py --source ./sample-docs/

# Start the API
uvicorn src.api.app:app --reload

# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the data retention policy?", "user_id": "user@company.com"}'
```

---

## Evaluation

This project includes a RAG evaluation harness using the RAGAS framework measuring:

- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are the retrieved chunks relevant?
- **Context Recall** — are all relevant chunks being retrieved?

---

## Why This Matters for Enterprise AI

Most enterprise AI projects fail not because of the model but because of the data layer beneath it. This architecture reflects lessons from real enterprise deployments: the access control problem is harder than the retrieval problem; chunking strategy matters more than model choice; and governance logging is not optional when dealing with regulated industries.

---

## Related Projects

- [AI Governance Framework](../02-ai-governance-framework) — the policy layer that governs this platform
- [Azure Data Lakehouse Architecture](../03-azure-data-lakehouse) — the upstream data platform feeding document stores
- [Enterprise Data Catalog](../05-data-catalog-metadata-engine) — metadata lineage for ingested documents
