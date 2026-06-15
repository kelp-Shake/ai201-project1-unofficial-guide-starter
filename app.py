"""
The Unofficial Guide - Milestone 5: Query Interface (Gradio)

A minimal web UI over the RAG pipeline. Type a question about UMD alumni
resources; the app retrieves relevant chunks, generates a grounded answer with
Groq, and shows which source document(s) the answer was drawn from.

Run:  python3 app.py     then open http://localhost:7860
"""
import gradio as gr

from query import ask


def handle_query(question):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    sources = result["sources"]
    sources_md = "\n".join(f"• {s}" for s in sources) if sources else "—"
    return result["answer"], sources_md


EXAMPLES = [
    "Is there a networking website for alumni?",
    "What can alumni do for career coaching?",
    "Can I use the library late at night after graduation?",
    "What membership tiers are available?",
]

with gr.Blocks(title="The Unofficial Guide — UMD Alumni") as demo:
    gr.Markdown(
        "# The Unofficial Guide — UMD Alumni\n"
        "Ask about alumni resources, networking, coaching, library access, and "
        "membership. Answers come **only** from the collected documents; if the "
        "documents don't cover your question, the assistant will say so."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. Is there a networking website for alumni?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from (sources)", lines=4)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
