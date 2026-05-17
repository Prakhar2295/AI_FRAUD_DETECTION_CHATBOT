# Banking Fraud Detection Voice AI - Phase 7

Phase 7 adds persistent fraud memory, semantic retrieval, adaptive risk enrichment, and cross-session fraud intelligence.

```text
Realtime Audio Stream
-> Fraud Audio Intelligence
-> Behavioral Fraud Intelligence
-> Persistent Fraud Memory Layer
   -> Session Memory
   -> Vector Fraud Memory
   -> Fraud Pattern Retrieval
   -> Adaptive Risk Correlation
   -> Cross-Session Intelligence
   -> Fraud Knowledge Enrichment
-> LangGraph Fraud Workflow
-> Adaptive Conversational Fraud Intelligence
-> Spoken AI Response
```

This phase focuses only on memory, retrieval, and adaptive intelligence. It does not add frontend UI, Docker, cloud deployment, federated learning, distributed training, Redis, or autonomous self-training.

## Architecture Additions

```text
backend/app/
├── memory/
│   ├── session_memory.py
│   ├── vector_memory.py
│   ├── fraud_memory_manager.py
│   ├── retrieval_engine.py
│   ├── adaptive_risk_engine.py
│   ├── fraud_pattern_store.py
│   ├── memory_pipeline.py
│   ├── memory_coordinator.py
│   └── conversation_history_manager.py
├── vectorstore/
│   ├── chroma_manager.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── fraud_similarity_search.py
│   └── vector_index_manager.py
├── knowledge/
│   ├── fraud_knowledge_base.py
│   ├── fraud_pattern_registry.py
│   ├── risk_enrichment_engine.py
│   └── adaptive_learning_service.py
└── graph/nodes/
    ├── memory_node.py
    ├── retrieval_node.py
    ├── adaptive_risk_node.py
    └── response_node.py
```

## Setup

From the project root:

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
```

Phase 7 direct dependencies:

```text
chromadb==1.5.9
sentence-transformers==5.5.0
```

The default embedding provider is deterministic and fully local:

```bash
export EMBEDDING_PROVIDER=deterministic
```

To opt into sentence-transformers embeddings:

```bash
export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

The sentence-transformers mode may download model weights the first time it runs. The deterministic provider avoids that for low-latency local development.

## Vector Memory Configuration

```bash
export CHROMA_PERSIST_DIR=backend/data/chroma
export CHROMA_COLLECTION_NAME=fraud_memory
export EMBEDDING_DIMENSIONS=128
export RETRIEVAL_TOP_K=4
export RETRIEVAL_SIMILARITY_THRESHOLD=0.55
```

ChromaDB is used when available. If Chroma fails to initialize, the system falls back to an in-memory vector store so the realtime workflow can continue.

## Retrieval Testing

Run a local memory and retrieval smoke test:

```bash
venv/bin/python -c 'import sys, tempfile, os; sys.path.insert(0,"backend"); os.environ["CHROMA_PERSIST_DIR"]=tempfile.mkdtemp(); from app.services.memory_service import MemoryService; from app.models.conversation_models import ConversationInteraction; from app.models.response_models import FraudSignalAnalysisResponse, RiskAnalysisResponse; from app.graph.state import utc_now_iso; m=MemoryService(); i=ConversationInteraction(timestamp=utc_now_iso(), transcript="Caller demanded OTP immediately to unblock account", fraud=FraudSignalAnalysisResponse(suspicious_indicators=["OTP request","urgency"], urgency_manipulation=True, emotional_pressure=True, suspicious_intent=True, llm_reasoning="urgent OTP demand"), risk=RiskAnalysisResponse(fraud_risk_score=85, risk_level="high", reasoning_summary="high risk")); m.append_interaction("prior-session", i); r=m.retrieve_similar_fraud("current-session", "urgent OTP request to unblock bank account"); print(r["semantic_retrieval_metadata"]); print(r["retrieved_fraud_patterns"][:1])'
```

Expected result:

- at least one retrieved fraud pattern
- metadata containing vector hit count, pattern hit count, and threshold
- retrieved metadata with prior `risk_level` and `fraud_risk_score`

## Adaptive Fraud Testing Scenario

Seed a prior high-risk OTP interaction, then run a new transcript containing similar urgency and credential pressure. The workflow should:

- retrieve the prior scenario
- add semantic retrieval metadata
- apply adaptive risk enrichment
- include `retrieval_node` and `adaptive_risk_node` in the workflow trace

## LangGraph Integration

The workflow now includes memory-aware nodes:

```text
intent
-> behavioral
-> retrieval
-> fraud
-> risk
-> adaptive_risk
-> response
-> memory
-> END
```

`retrieval_node` enriches workflow state with:

- `retrieved_fraud_patterns`
- `semantic_retrieval_metadata`
- `historical_fraud_context`
- `fraud_knowledge_context`

`adaptive_risk_node` enriches workflow state with:

- adjusted risk score
- adjusted risk level
- `adaptive_risk_enrichment`
- cross-session correlation metadata

`memory_node` persists the completed interaction into session memory and vector memory.

## Design Notes

### Memory Architecture Decisions

`MemoryService` is the facade used by graph nodes and realtime orchestration. Internally it delegates to `FraudMemoryManager`, `MemoryCoordinator`, `SessionMemory`, and vector retrieval services. This keeps memory independent from WebSocket transport, TTS playback, and frontend concerns.

### Vector Retrieval Strategy

Fraud interactions are embedded and stored with risk metadata. Retrieval uses semantic vector similarity plus rule-based pattern matching from `FraudPatternStore`.

### Adaptive Risk Enrichment

Adaptive scoring does not self-train. It applies deterministic enrichment based on retrieved similar scenarios and session history. This gives memory-assisted risk correlation without autonomous model updates.

### Fraud Similarity Search

`FraudSimilaritySearch` filters vector hits by configurable threshold. This gives deterministic retrieval behavior and keeps future GPU/vector provider swaps isolated.

### Long-Term Memory Strategy

Session memory tracks conversation continuity. Chroma-backed vector memory supports cross-session fraud recall. The current interfaces are designed so Redis, graph databases, or distributed vector stores can replace internals later.

### Future Paths

- Graph database integration: map fraud interactions, callers, scam archetypes, and risk transitions as graph nodes and edges.
- Distributed retrieval: replace local Chroma with a network vector service behind `ChromaManager`/`RetrievalService`.
- Autonomous learning: add human-reviewed pattern promotion before any self-updating fraud knowledge.
- Observability: add retrieval latency, embedding latency, vector hit quality, memory write failures, and adaptive score deltas.

### Potential Bottlenecks

- Sentence-transformers model loading and embedding latency when enabled.
- Chroma disk I/O as memory grows.
- Ollama latency during graph nodes.
- Retrieval quality with deterministic embeddings; use sentence-transformers for stronger semantic matching once model weights are available locally.

