# RAG Database Design (Medical Agent)

## 1) Knowledge Sources

- Clinical guidelines (disease triage and first-visit recommendations)
- Drug knowledge:
  - indications
  - contraindications
  - interactions
  - dosage range notes
- Medical education content for patient-friendly explanation
- Local service process (hospital departments, online/offline routing)

## 2) Storage Layout

Suggested path:
- Raw docs: `/root/autodl-tmp/medagent/datasets/rag_raw`
- Processed chunks: `/root/autodl-tmp/medagent/rag/chunks`
- Vector index: `/root/autodl-tmp/medagent/rag/index`

Each chunk metadata:
- `source_id`
- `source_type` (`guideline`/`drug`/`education`/`process`)
- `disease_tags`
- `drug_tags`
- `updated_at`
- `evidence_level`

## 3) Retrieval Pipeline

1. Query classifier (triage / medication / report interpretation)
2. Hybrid retrieval:
   - BM25 sparse retrieval
   - dense embedding retrieval
3. Rerank top-K chunks
4. Evidence-constrained generation:
   - final answer must include citation IDs

## 4) Embedding and Index

Starter:
- Embedding model: `BAAI/bge-m3` (multilingual)
- Vector DB options:
  - Milvus (recommended for production)
  - FAISS (lightweight prototype)

## 5) Quality and Safety Gates

- Citation coverage ratio (`answers with references / all answers`)
- Source freshness checks
- Medical safety checker:
  - emergency symptom detection
  - contraindication warning coverage

## 6) Benchmark Metrics

- Grounding: citation hit rate
- Medical correctness: expert rubric or Critic model
- Safety: high-risk recall (do-not-miss cases)
- UX: multi-turn completion and average turns

