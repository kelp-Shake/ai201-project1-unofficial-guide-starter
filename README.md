# The Unofficial Guide — Project 1

A retrieval-augmented (RAG) question-answering system for **University of Maryland
alumni**. Ask a question about alumni resources and the system retrieves relevant
passages from a collection of Reddit threads, LinkedIn, and official UMD Alumni
Association pages, then generates an answer grounded *only* in those passages,
with source attribution.

**Pipeline:** Ingestion → Chunking (LangChain) → Embedding (all-MiniLM-L6-v2) →
Storage + Retrieval (ChromaDB) → Generation (Groq llama-3.3-70b-versatile) → Output

**Run it:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Groq API key
python chunk_pipeline.py      # build chunks.json
python embed_store.py         # embed + load ChromaDB, prints retrieval eval
python app.py                 # launch the Gradio UI at http://localhost:7860
```

---

## Domain

This system covers **resources and communities for University of Maryland alumni** —
career coaching, networking platforms, membership benefits, library/campus access,
and what alumni life looks like after graduation.

This knowledge is valuable because the post-graduation transition from student to
working professional is hard, and a support network plus access to university
resources makes it easier. It is hard to find through official channels because the
information is scattered across many platforms: official Alumni Association pages,
LinkedIn, and informal first-hand accounts on Reddit. The official pages tell you
what the association *offers*; the Reddit threads tell you what actually works in
practice (e.g., that alumni can use libraries only during public hours, not late
night). A single guidebook that combines both perspectives doesn't exist.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/UMD — "Alumni, your terpmail may be deleted if you don't take action" | Reddit thread | https://www.reddit.com/r/UMD/comments/1dpc8wj/ |
| 2 | r/UMD — "UMD Alumni Association: What's In It For Me?" | Reddit thread | https://www.reddit.com/r/UMD/comments/1cr6efr/ |
| 3 | r/UMD — "I hate UMD. Stop sending me alumni mail." | Reddit thread | https://www.reddit.com/r/UMD/comments/1ff0uor/ |
| 4 | r/UMD — "What libraries and spaces are available for me, an alumni" | Reddit thread | https://www.reddit.com/r/UMD/comments/1i3oo33/ |
| 5 | r/UMD — "How weird would it be as an alumni to just hang around campus" | Reddit thread | https://www.reddit.com/r/UMD/comments/1kf0siy/ |
| 6 | University of Maryland Alumni Association — LinkedIn | LinkedIn page (.txt) | https://www.linkedin.com/company/maryland-alumni |
| 7 | UMD Alumni Association — Coaches Corner | Official webpage (.md) | https://alumni.umd.edu/resources/coaches-corner |
| 8 | UMD Alumni Association — Corporate Connections | Official webpage (.md) | https://alumni.umd.edu/resources/corporate-engagement |
| 9 | UMD Alumni Association — Membership Benefits | Official webpage (.md) | https://alumni.umd.edu/membership/benefits |
| 10 | UMD Alumni Association — Terrapins Connect | Official webpage (.md) | https://alumni.umd.edu/resources/terrapins-connect |

Documents were captured as `.md`/`.txt` via a browser extension and stored in
`documents/`.

---

## Chunking Strategy

The corpus has two structurally different document types, so chunking is
**two-track** and structure-aware (`chunk_pipeline.py`). Size-based sub-splitting
uses LangChain's `RecursiveCharacterTextSplitter` on sentence/whitespace
separators.

**Reddit threads:**
- One chunk for `title + OP body`, then **one chunk per comment**.
- A comment is kept whole unless it exceeds a **500-character cap**, in which case
  it is sub-split with ~30-char (~6%) overlap.
- Low overlap is deliberate: comments are independent turns in a conversation, so
  there is little semantic continuity to preserve across boundaries.

**Official pages / LinkedIn:**
- Split on markdown headers (`#`/`##`/`###`) into sections, each capped at **500
  characters** with **75-char (~15%) overlap**.
- Preprocessing strips markdown images and link URLs (keeping anchor text),
  collapses whitespace, and trims repeated site-chrome footers (the "Career
  Resources"/"Contact Options" nav block).
- Section headers are prepended to each chunk (`"{header}. {body}"`) so a chunk
  carries its topic label into the embedding.

Fragments shorter than 40 characters are dropped.

**Chunk size:** 500-character cap for both tracks.

**Overlap:** ~6% (Reddit) / ~15% (pages).

**Why these choices fit your documents:** Reddit comments run roughly 400–900
characters, so a 500-char cap usually keeps a comment whole while preventing the
longest ones from dominating; pages are paragraph-structured under headers, where
~15% overlap preserves continuity across a section split.

**Final chunk count:** **339** chunks total (205 Reddit, 134 page/LinkedIn) across
11 documents.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim). Chosen
because it runs locally with no API key and no rate limits, is widely used and
well-reviewed, and is fast enough to embed the whole corpus in seconds. Embeddings
are normalized and stored in ChromaDB with the collection configured for **cosine
distance** (`hnsw:space="cosine"`) so scores fall on the 0–2 cosine scale the
project guidance assumes (Chroma's default is squared-L2, whose numbers would not
line up with the "below 0.5" relevance heuristic).

**Production tradeoff reflection:** If I were deploying this for a larger audience —
post-grads from many universities, majors, and backgrounds — I would weigh: (1)
**domain accuracy**, since MiniLM is general-purpose and a larger or instruction-
tuned embedding model would better separate near-duplicate alumni posts; (2)
**multilingual support**, for a more diverse user base; (3) **context length**, so
longer page sections embed without truncation; and (4) **latency vs. local
hosting**, where an API-hosted model removes the need to ship model weights but adds
per-query cost and network dependence. With cost no object I would likely move to a
hosted proprietary embedding API (e.g., Google Gemini) to cut operational overhead
and gain accuracy on domain-specific text.

---

## Grounded Generation

Generation runs through `query.py`'s `ask()` function: it retrieves the top-5
chunks, formats them into a numbered, source-labeled `CONTEXT` block, and sends them
to Groq's `llama-3.3-70b-versatile` at low temperature (0.1).

**System prompt grounding instruction:** The model is told the context is the *only*
permitted source, with an exact decline string for the no-coverage case (not a
vague suggestion):

> "You answer questions using ONLY the information in the CONTEXT documents…
> 1. Use only facts stated in the CONTEXT. Do not use any outside or prior
> knowledge, and do not guess or infer beyond what the text says.
> 2. If the CONTEXT does not contain enough information to answer, reply with
> exactly: *"I don't have enough information on that."* and nothing else."

Low temperature biases the model toward faithful extraction rather than fluent
invention.

**How source attribution is surfaced in the response:** Attribution is
**programmatic, not model-generated**. After generation, `ask()` collects the unique
`source` filenames from the retrieved chunks' metadata and returns them as a
`sources` list, which the Gradio UI renders in a separate "Retrieved from" panel.
Sources are attached **only when the model actually answered** — if it returns the
decline string, the `sources` list is empty, so a non-answer is never falsely
attributed.

---

## Evaluation Report

All five questions were run verbatim from `planning.md` through `ask()`.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What resources can alumni use on campus | Public spaces | Libraries and quiet study spaces, plus mentorship/volunteer opportunities via the Alumni Association | Relevant | Accurate |
| 2 | Is there alumni I connect with for coaching for my career goals | Yes, career coaches | Yes — Coaches Corner program and the Terp Referral Exchange Business Directory | Relevant | Accurate |
| 3 | Is there a networking website for alumni | Yes, Terrapins Connect | Yes — alumni.umd.edu and a platform called Terrapins Connect | Relevant | Accurate |
| 4 | What type of membership length periods are available | 1 year, 3 years, lifetime | "I don't have enough information on that." | Off-target | **Inaccurate** |
| 5 | Can I use the library late night after graduation | No, only during public hours | No — only when publicly available; no late-night McKeldin | Partially relevant | Accurate |

**Score: 4/5 accurate, 1/5 inaccurate (Q4).** Distance scores on the top results
for Q1–Q3 were all below 0.5 (0.43, 0.35, 0.40); Q5's answer chunk was retrieved at
rank 3 (0.55); Q4's answer chunk was never retrieved (best result 0.62). Full
transcript in `eval_results.txt`.

---

## Failure Case Analysis

**Question that failed:** Q4 — "What type of membership length periods are
available?" (expected: 1 year / 3 years / lifetime).

**What the system returned:** "I don't have enough information on that." — a decline,
even though the answer *is* present in the corpus.

**Root cause (tied to a specific pipeline stage):** This is a **chunking +
embedding** failure, not a generation failure. The answer lives in a markdown table
on the Membership Benefits page, where the membership tiers are the *column headers*
of the table: `| Benefits | Life | Three Year | Annual |`. During chunking that
table became one chunk dominated by ~15 benefit-row lines and pipe (`|`) characters,
so the three tier words are a tiny fraction of the chunk's text. When that chunk is
embedded, its vector is pulled toward "list of member benefits," not "membership
durations." As a result, for the query *"membership length periods"* the chunk
scores ~0.62+ and **falls outside the top-5 retrieval cutoff** — the chunks that do
get retrieved (a "Membership / Renew / Benefits" nav fragment and a Reddit comment
"No you can purchase a membership as well") don't contain the tiers. Generation then
behaves *correctly*: the answer genuinely isn't in its context, so the grounding
rule makes it decline rather than hallucinate. The grounding worked; retrieval
starved it.

**What you would change to fix it:** Improve how tabular content is chunked — detect
markdown tables and either (a) keep the header row attached to each data row so the
tier names co-occur with each benefit, or (b) linearize the table into sentences
("The Lifetime, Three Year, and Annual memberships include…") so the tier vocabulary
isn't drowned out by formatting. Raising `top_k` would also pull the chunk into
context, at the cost of diluting other queries. The cleanest fix is at the chunking
stage, since the information density of that chunk is the real problem.

---

## Spec Reflection

**One way the spec helped you during implementation:** The `planning.md` Chunking
Strategy section committed me to a *two-track, structure-aware* approach before I
wrote any code, with concrete numbers (500-char cap, low Reddit overlap, ~15% page
overlap). That meant the implementation had explicit targets to hit and a clear
reason to write separate `parse_reddit()` and `parse_page()` paths instead of one
generic splitter. The architecture diagram likewise fixed the tool at each stage
(LangChain → MiniLM → ChromaDB → Groq), so wiring the pipeline together was a matter
of following the spec rather than re-deciding components mid-build.

**One way your implementation diverged from the spec, and why:** The plan originally
set the Reddit comment cap at **200 characters**, but inspecting real chunks showed
comments run 400–900 characters and a 200-cap was fragmenting them mid-sentence and
inflating the chunk count with low-signal pieces. I raised the cap to **500** and
updated `planning.md` to record the change. A second divergence: the spec's
production reflection names Gemini as the long-term embedding choice, but for this
build I used the recommended local `all-MiniLM-L6-v2` because it needs no API key or
budget — the Gemini note remains a *future* consideration, not what's wired up.

---

## AI Usage

**Instance 1 — Chunking pipeline**

- *What I gave the AI:* My `planning.md` Chunking Strategy section and the
  architecture diagram, asking it to implement the ingestion + cleaning + chunking
  stage for the two document types.
- *What it produced:* `chunk_pipeline.py` with separate Reddit/page parsers, the
  LangChain `RecursiveCharacterTextSplitter`, and a dependency-free fallback splitter
  wrapped in a `try/except` so it would run before installing anything.
- *What I changed or overrode:* After inspecting real output I bumped the Reddit cap
  from 200 → 500 (comments were fragmenting), then removed the `try/except` fallback
  so LangChain is a hard, explicit dependency rather than silently degrading. I also
  had it delete the resulting dead code (`title_from`, unused `NAV_MARKERS`, stray
  loop variables) and verified the chunk count stayed at 339.

**Instance 2 — Grounded generation + retrieval tuning**

- *What I gave the AI:* My grounding requirement (answers from retrieved context
  only, with source attribution) and the output format I wanted (answer + source
  list), asking it to wire retrieval to the Groq LLM and build the interface.
- *What it produced:* `query.py` with a context-only system prompt and `embed_store.py`
  storing chunks in ChromaDB, plus an `app.py` Gradio UI.
- *What I changed or overrode:* I directed two corrections. First, source
  attribution had to be **programmatic** (derived from retrieved-chunk metadata and
  only attached when the model answered), not left to the LLM to add. Second, I had
  the ChromaDB collection switched to **cosine distance** because the default
  squared-L2 scores didn't match the project's "below 0.5" relevance guidance. I
  also overrode the interface dependency, pinning `gradio>=5,<6` after gradio 6
  pulled a `huggingface-hub` version that broke `sentence-transformers`.
