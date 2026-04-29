"""
retriever.py — BA Agent Search & Answer Engine
------------------------------------------------
Searches ChromaDB for relevant document chunks,
optionally searches the web via Tavily,
then sends everything to Claude to generate
a cited, conversational answer.

Usage:
    python3 retriever.py
"""

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import anthropic

load_dotenv()  # reads your .env file

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
CHROMA_FOLDER   = "chroma_db"
COLLECTION_NAME = "ba_documents"
TOP_K_RESULTS   = 5          # how many chunks to retrieve per question
MAX_HISTORY     = 10         # how many past messages to remember

# ─────────────────────────────────────────
# STEP 1 — Connect to ChromaDB memory
# ─────────────────────────────────────────

def load_collection():
    """Connect to the stored document memory."""
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    client       = chromadb.PersistentClient(path=CHROMA_FOLDER)

    try:
        collection = client.get_collection(
            name               = COLLECTION_NAME,
            embedding_function = embedding_fn
        )
        print(f"  📚 Memory loaded — {collection.count()} chunks available\n")
        return collection
    except Exception:
        print("\n  ❌ No document memory found.")
        print("  👉 Run python3 ingest.py first to load your documents.\n")
        return None


# ─────────────────────────────────────────
# STEP 2 — Search documents for relevant chunks
# ─────────────────────────────────────────

def search_documents(collection, question):
    """Find the most relevant chunks from uploaded documents."""
    results = collection.query(
        query_texts = [question],
        n_results   = TOP_K_RESULTS
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text"    : results["documents"][0][i],
            "source"  : results["metadatas"][0][i].get("source", "Unknown"),
            "location": results["metadatas"][0][i].get("location", "")
        })

    return chunks


# ─────────────────────────────────────────
# STEP 3 — Search the web (optional)
# ─────────────────────────────────────────

def search_web(question):
    """Search the web for additional context using Tavily."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return []

    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=tavily_key)
        results = client.search(query=question, max_results=3)
        web_chunks = []
        for r in results.get("results", []):
            web_chunks.append({
                "text"    : r.get("content", ""),
                "source"  : r.get("url", "Web"),
                "location": "Web Search"
            })
        return web_chunks
    except Exception as e:
        print(f"  ⚠️  Web search unavailable: {e}")
        return []


# ─────────────────────────────────────────
# STEP 4 — Build the prompt with context
# ─────────────────────────────────────────

def build_context(doc_chunks, web_chunks):
    """Format retrieved chunks into a readable context block for Claude."""
    context_parts = []

    if doc_chunks:
        context_parts.append("=== FROM YOUR UPLOADED DOCUMENTS ===\n")
        for i, chunk in enumerate(doc_chunks, start=1):
            context_parts.append(
                f"[{i}] Source: {chunk['source']} | {chunk['location']}\n"
                f"{chunk['text']}\n"
            )

    if web_chunks:
        context_parts.append("\n=== FROM WEB SEARCH ===\n")
        for i, chunk in enumerate(web_chunks, start=len(doc_chunks)+1):
            context_parts.append(
                f"[{i}] Source: {chunk['source']}\n"
                f"{chunk['text']}\n"
            )

    return "\n".join(context_parts)


# ─────────────────────────────────────────
# STEP 5 — Ask Claude to generate an answer
# ─────────────────────────────────────────

def ask_claude(question, context, conversation_history):
    """Send question + context + history to Claude and get a cited answer."""

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are an expert Business Analyst assistant.
You help Business Analysts understand their project documents by answering
questions clearly, accurately, and with citations.

RULES:
1. Always answer based on the provided context first (uploaded documents).
2. If the context does not have enough information, say so clearly.
3. Always cite your sources using the format: [Source: filename | location]
4. Keep answers clear, structured, and professional.
5. If asked for a summary, provide bullet points with key findings.
6. Remember the conversation history and refer back to it when relevant.
7. If you find conflicting information across documents, flag it clearly.

FORMAT your answers like this:
- Lead with a direct answer
- Support with evidence from the documents
- End with citations
- If relevant, suggest follow-up questions the BA might want to ask"""

    # Build messages with full conversation history
    messages = conversation_history.copy()

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Context from documents and web search:
{context}

---
Question: {question}

Please answer based on the context above. Cite your sources clearly."""
    })

    response = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 1500,
        system     = system_prompt,
        messages   = messages
    )

    return response.content[0].text


# ─────────────────────────────────────────
# STEP 6 — Format the final answer nicely
# ─────────────────────────────────────────

def format_sources(doc_chunks, web_chunks):
    """Print a clean list of all sources used."""
    all_sources = []

    for c in doc_chunks:
        all_sources.append(f"  📄 {c['source']} — {c['location']}")

    for c in web_chunks:
        all_sources.append(f"  🌐 {c['source']}")

    if all_sources:
        print("\n  Sources consulted:")
        for s in set(all_sources):  # deduplicate
            print(s)


# ─────────────────────────────────────────
# MAIN — Conversational loop
# ─────────────────────────────────────────

def chat():
    print("\n" + "="*55)
    print("  BA Agent — Document Q&A (type 'exit' to quit)")
    print("="*55)

    # Load document memory
    collection = load_collection()
    if not collection:
        return

    conversation_history = []  # remembers the full conversation

    while True:
        print()
        question = input("  You: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit", "bye"):
            print("\n  👋 BA Agent session ended. Goodbye!\n")
            break

        print("\n  🔍 Searching documents...")

        # Search documents
        doc_chunks = search_documents(collection, question)

        # Search web (optional — comment this out if you don't have Tavily key)
        use_web = os.getenv("TAVILY_API_KEY") is not None
        web_chunks = search_web(question) if use_web else []

        if use_web and web_chunks:
            print(f"  🌐 Found {len(web_chunks)} web results")

        # Build context
        context = build_context(doc_chunks, web_chunks)

        print("  🤖 Generating answer...\n")

        # Get answer from Claude
        answer = ask_claude(question, context, conversation_history)

        # Print the answer
        print("  " + "-"*50)
        print(f"\n  Agent:\n")
        # Indent each line for readability
        for line in answer.split("\n"):
            print(f"  {line}")

        # Show sources used
        format_sources(doc_chunks, web_chunks)

        print("\n  " + "-"*50)

        # Save to conversation history (so agent remembers)
        conversation_history.append({"role": "user",      "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        # Keep history to last N messages to avoid token overflow
        if len(conversation_history) > MAX_HISTORY * 2:
            conversation_history = conversation_history[-(MAX_HISTORY * 2):]


if __name__ == "__main__":
    chat()
