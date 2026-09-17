# VIC Road Safety RAG (Walert-style reproduction)

Test-driven RAG assistant for Victorian road-safety Q&A, with driver-stage
(learner/P1/P2/full) and topic-category precision as the core differentiator.
Reproduces the pipeline architecture of RMIT's Walert project
(github.com/rmit-ir/walert) - a fresh implementation, not a fork, adapted to
a different domain (long-form road rules vs FAQ pairs).

## WIL Project Group-77

Rithika Pamu - s4146941,
Pranav Patil - s4229967,
Rujuta Patil - s4189237,
Dev Jadeja   - s4217956,
Vivek Sharma - s4233203,
Mohammad Mahin Shekh - s4220857

## Project structure

```
roadsafety-rag/
├── data/
│   ├── raw/                 # source PDFs/HTML/txt - name files to match
│   │                          doc_id in sources.csv for provenance tracking
│   ├── sources.csv          # Phase 1: provenance log (url, date_accessed,
│   │                          driver_stage, topic) per source document
│   ├── personas/
│   │   └── personas.json    # Phase 1: 3 personas + example questions
│   ├── processed/           # chunked_docs.jsonl gets written here
│   └── chroma_db/           # vector store (gitignored)
├── src/
│   ├── ingestion.py          # parse PDF/HTML/txt, attach provenance
│   ├── chunking.py           # chunk + tag driver_stage/topic
│   ├── embed_store.py        # sentence-transformers + Chroma
│   ├── retrieval.py          # query + stage filtering
│   ├── generation.py         # Ollama prompt + confidence guardrail
│   └── pipeline.py           # CLI: build / ask
├── eval/
│   ├── test_set_template.csv # Phase 3: labelled test questions (persona,
│   │                          expected stage/topic, gold source, answerable?)
│   ├── evaluate.py            # quick doc-level retrieval accuracy check
│   └── evaluate_full.py       # Phase 3: full metric suite - effectiveness,
│                                stage/category precision, source
│                                attribution, faithfulness (manual flag)
├── docs/
│   ├── business_case.md      # Phase 1: 1-paragraph business case template
│   ├── report_outline.md     # Phase 4: technical report skeleton
│   └── team_process.md       # branch conventions, ownership split, check-ins
└── app.py                    # Streamlit UI
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

## steps to run model 
pip install -r requirements.txt
ollama pull llama3.2:3b
 
```
## Usage

```bash
python src/pipeline.py build-collection
python src/pipeline.py ask "Can a P1 driver use a hands-free phone?"
streamlit run app.py
```

## Workflow, mapped to the project checklist

**Phase 1 (Week 1-2):**
1. Collect real VIC source docs (VicRoads handbook, learner logbook guide,
   P1/P2 conditions, TAC stats, common FAQs) into `data/raw/`.
2. Log each one as a row in `data/sources.csv` - filename stem must match
   the `doc_id` column so provenance flows through automatically.
3. Fill in `data/personas/personas.json` with real example questions per
   persona (a starter set is already there).
4. Write the business case in `docs/business_case.md`.

**Phase 2 (Week 2-4):** already implemented in `src/` - see inline comments
in `chunking.py` for what to experiment with (chunk size) and
`generation.py` for the guardrail threshold.

```bash
python src/pipeline.py build
python src/pipeline.py ask "Can a P1 driver use a hands-free phone?"
streamlit run app.py
```

**Phase 3 (Week 4-6, alongside Phase 2):**
1. Fill in `eval/test_set_template.csv` with real labelled questions
   (persona, expected driver_stage/topic, gold source doc, whether it
   should be answerable at all - include some deliberately out-of-scope
   questions to test abstention).
2. Run the full suite after every meaningful pipeline change:
   ```bash
   python eval/evaluate_full.py
   ```
   This writes a timestamped per-question CSV to `eval/` and prints a
   summary (effectiveness breakdown, stage/topic precision, source
   attribution accuracy). Fill in the blank `faithful` column by hand -
   faithfulness (does the answer match what the source says) isn't
   automatable with this stack without adding an NLI/LLM-judge step.
3. Keep old `results_*.csv` files - they're your before/after evidence for
   Phase 4.

**Phase 4 (Week 6-8):** work through `docs/report_outline.md` section by
section, pulling in the business case, architecture, and eval results
you've already produced.

**Team process:** see `docs/team_process.md` for branch conventions,
ownership split, and the weekly check-in template.
