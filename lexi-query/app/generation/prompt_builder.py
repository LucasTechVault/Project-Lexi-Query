def build_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        meta = c["metadata"]
        blocks.append(
            f"[doc={meta['doc_id']} page={meta['page_num']} chunk={meta['chunk_index']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)

def build_messages(question: str, chunks: list[dict]):
    context = build_context(chunks)
    system = (
        "You are a financial QA assistant. "
        "Answer only from the provided context. "
        "If the answer is not supported by the context, say you cannot determine it. "
        "Cite the supporting doc and page in your answer."
    )
    user = f"""Question:
{question}

Retrieved context:
{context}

Instructions:
- Give a concise, direct answer.
- Then provide a short evidence section.
- Cite as [doc_id p.X].
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]