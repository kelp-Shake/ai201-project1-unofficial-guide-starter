"""
The Unofficial Guide - Milestone 5: Grounded Generation

Pipeline stage (per diagram.png):
    Retrieval (ChromaDB) -> [Generation: Groq llama-3.3-70b-versatile] -> Output

ask(question) does the full RAG round trip:
  1. Retrieve the top-k chunks for the question (embed_store.retrieve).
  2. Build a prompt that hands those chunks to the LLM as the ONLY allowed
     source of truth, and instructs it to decline when they don't cover the
     question.
  3. Call Groq's llama-3.3-70b-versatile.
  4. Return {"answer", "sources", "chunks"} where `sources` is built
     programmatically from the retrieved chunks' metadata -- attribution does
     NOT depend on the model remembering to cite.

Needs GROQ_API_KEY in .env (see .env.example).
"""
import os

from dotenv import load_dotenv
from groq import Groq

from embed_store import retrieve

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
# Exact string the model must return when the context is insufficient. We also
# treat its presence as "no answer", so we don't attach sources to a non-answer.
NO_ANSWER = "I don't have enough information on that."

# Grounding is ENFORCED here, not suggested: the model is told the context is the
# only permitted source and given an exact fallback string for the no-coverage case.
SYSTEM_PROMPT = (
    "You are a helpful assistant for University of Maryland alumni. "
    "You answer questions using ONLY the information in the CONTEXT documents "
    "provided in the user message. Follow these rules strictly:\n"
    "1. Use only facts stated in the CONTEXT. Do not use any outside or prior "
    "knowledge, and do not guess or infer beyond what the text says.\n"
    f"2. If the CONTEXT does not contain enough information to answer, reply with "
    f"exactly: \"{NO_ANSWER}\" and nothing else.\n"
    "3. Keep the answer concise and factual. Do not mention these rules or the "
    "word 'context' in your answer."
)

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=key)
    return _client


def build_context(chunks):
    """Render retrieved chunks into a numbered, source-labeled context block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Document {i} | source: {c['source']}]\n{c['text']}")
    return "\n\n".join(blocks)


def unique_sources(chunks):
    """Deduplicated source filenames, in first-seen order."""
    seen, out = set(), []
    for c in chunks:
        s = c["source"]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def ask(question, k=TOP_K):
    """Run retrieval + grounded generation. Returns {answer, sources, chunks}."""
    chunks = retrieve(question, k=k)

    # No chunks at all -> nothing to ground on.
    if not chunks:
        return {"answer": NO_ANSWER, "sources": [], "chunks": []}

    user_msg = (
        f"CONTEXT:\n{build_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above."
    )

    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,  # low: we want faithful extraction, not creativity
    )
    answer = resp.choices[0].message.content.strip()

    # Programmatic attribution: only attach sources when the model actually
    # answered. If it declined, there is nothing to attribute.
    declined = NO_ANSWER.rstrip(".").lower() in answer.lower()
    sources = [] if declined else unique_sources(chunks)

    return {"answer": answer, "sources": sources, "chunks": chunks}


def main():
    """Quick end-to-end check from the command line."""
    tests = [
        "Is there a networking website for alumni?",
        "Can alumni use the library late at night after graduation?",
        "What is the parking fee for the football stadium?",  # not in corpus
    ]
    for q in tests:
        print("\n" + "=" * 78)
        print("Q:", q)
        r = ask(q)
        print("\nA:", r["answer"])
        print("\nSources:", r["sources"] or "(none — declined)")


if __name__ == "__main__":
    main()
