"""Generation module — build a grounded prompt and call the local LLM.

Shared by every RAG variant (Naive_rag.py, parent_child_rag.py). Keeps the
generator independent of how chunks were retrieved: callers pass retrieved
(score, source, text) triples, this module handles prompt + Ollama.
"""

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5"

EMBED_MODEL = "all-MiniLM-L6-v2"   # dense bi-encoder, 384 dims, local; shared by every RAG variant


def build_prompt(question, retrieved):
    """Context-grounded prompt from retrieved (score, source, text) triples."""
    context = "\n\n".join(f"[{src}]\n{text}" for _, src, text in retrieved)
    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate(prompt):
    """Ask local Ollama. Returns (answer, None) on success, (None, reason) on failure.

    `think: false` disables reasoning models' slow chain-of-thought — qwen3.5
    answers in seconds instead of minutes, which is what caused earlier timeouts.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False,
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["response"].strip(), None
    except (TimeoutError, urllib.error.URLError) as e:
        # URLError wrapping a timeout still means "reachable but too slow"
        reason = e.reason if isinstance(e, urllib.error.URLError) else e
        if isinstance(reason, (TimeoutError, OSError)) and "timed out" in str(reason).lower():
            return None, f"'{OLLAMA_MODEL}' did not respond within 180s (model too slow)"
        return None, "Ollama not reachable at localhost:11434 (is `ollama serve` running?)"
    except OSError:
        return None, "Ollama not reachable at localhost:11434 (is `ollama serve` running?)"
