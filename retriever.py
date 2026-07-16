##############################################################################
# retriever.py
#
# PURPOSE:
#   This is stage 2 of the query-expansion RAG pipeline.
#   It loads the FAISS index that ingest.py built, wraps it in a
#   LangChain retriever object, and gives you a way to test raw
#   retrieval on its own — before query expansion is layered on top
#   in query_expansion.py.
#
# WHY THIS FILE EXISTS — THE BOUNDARY PRINCIPLE:
#   Think of this file as the single place that knows HOW we search
#   the vector store: which embedding model, which search type, how
#   many chunks to return per query. query_expansion.py should never
#   redefine any of that itself — it should just call get_retriever()
#   below and use whatever it gets back. That way there is exactly
#   one place to change "how many chunks do we retrieve," not two
#   places that can quietly drift apart.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   You can — running it directly lets you test plain retrieval (no
#   query expansion, no LLM answer) with a single question, and see
#   exactly which chunks come back. Useful for sanity-checking the
#   index before layering the rest of the pipeline on top.
#   ingest.py must already have been run at least once before this
#   will work, since it depends on the FAISS index existing on disk.
#
# HOW OTHER FILES USE THIS:
#   from retriever import get_retriever
#
#   retriever = get_retriever()
#   chunks = retriever.invoke("What was Q2 EBITDA?")
##############################################################################


import os
# os lets us check whether the FAISS index folder exists before
# trying to load it, and build source file paths for display.

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# WORKAROUND: on macOS, faiss and numpy/torch can each bundle their
# own copy of the OpenMP runtime (libomp.dylib). When both load in
# the same process, OpenMP aborts with "Error #15: already
# initialized" instead of silently picking one. This must be set
# BEFORE faiss is imported below. See the matching note in ingest.py.

from dotenv import load_dotenv
load_dotenv(override=True)
# Loads OPENAI_API_KEY from .env — needed here because turning a
# query into a vector still requires calling the embedding API,
# even though we are only searching, not ingesting.

from langchain_openai import OpenAIEmbeddings
# Same embedding model used in ingest.py. This has to match exactly —
# see the note on EMBEDDING_MODEL below.

from langchain_community.vectorstores import FAISS
# Loads the index files (index.faiss, index.pkl) that ingest.py saved.

from rich.table import Table
from rich.console import Console
from rich.panel import Panel
# rich formatting for readable terminal output — cosmetic only.

console = Console()


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "./faiss_index"
# Must match FAISS_INDEX_PATH in ingest.py exactly — this is where
# ingest.py saved the index, so this is where we load it from.

EMBEDDING_MODEL = "text-embedding-3-large"
# Must match EMBEDDING_MODEL in ingest.py exactly. A query vector
# and the document vectors it's compared against have to come from
# the same embedding model, or the similarity scores are meaningless —
# it's like comparing distances measured in miles against distances
# measured in kilometers without converting first.

TOP_K = 6
# How many chunks a single search returns. query_expansion.py will
# call this same retriever once per expanded query, so the total
# number of chunks considered across all expansions can be up to
# (number of query phrasings) x TOP_K, before deduplication.


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD THE VECTOR STORE FROM DISK
# ─────────────────────────────────────────────────────────────

def load_vector_store(index_path: str) -> FAISS:
    """
    Loads the previously saved FAISS index and its chunk store from disk.

    Args:
        index_path: Folder containing index.faiss and index.pkl,
            as saved by ingest.py.

    Returns:
        A FAISS vector store, ready to be searched.

    Raises:
        FileNotFoundError: If index_path doesn't exist. We raise here
        with a clear message pointing at ingest.py, rather than letting
        FAISS.load_local() fail later with a more cryptic file-not-found
        error deeper in its own code.
    """

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Please run ingest.py first to build the index."
        )

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = FAISS.load_local(
        folder_path=index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
        # This flag exists because loading a FAISS index involves
        # unpickling Python objects (index.pkl), and unpickling data
        # from an untrusted source can execute arbitrary code. It's
        # safe here because we only ever load an index we created
        # ourselves in ingest.py — never one from an external source.
    )

    console.print(f"FAISS index loaded from {index_path}")
    console.print(f"Vectors in index: {vector_store.index.ntotal}")
    return vector_store


# ─────────────────────────────────────────────────────────────
# STEP 2: WRAP THE VECTOR STORE IN A RETRIEVER
# ─────────────────────────────────────────────────────────────

def build_retriever(vector_store: FAISS):
    """
    Wraps a raw FAISS vector store in a LangChain retriever object.

    A vector store on its own only knows how to store and compare
    vectors. A retriever adds the standard `.invoke(query)` interface
    that the rest of LangChain — and query_expansion.py — expects.

    Args:
        vector_store: The FAISS store returned by load_vector_store().

    Returns:
        A LangChain retriever configured to return TOP_K chunks per query.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        # "similarity" = plain cosine-similarity nearest-neighbor search.
        # (Other options exist, like "mmr" for result diversity, but
        # we're keeping this simple and predictable for the demo.)
        search_kwargs={"k": TOP_K},
    )

    console.print(f"Retriever built: will return {TOP_K} chunks per query")
    return retriever


def get_retriever():
    """
    Convenience function that does both steps above in one call:
    load the index from disk, then wrap it as a retriever.

    This is the function query_expansion.py should import and call —
    it guarantees that file is always using the exact same retrieval
    configuration defined here, rather than rebuilding its own.

    Returns:
        A ready-to-use LangChain retriever.
    """

    vector_store = load_vector_store(FAISS_INDEX_PATH)
    return build_retriever(vector_store)


# ─────────────────────────────────────────────────────────────
# DISPLAY HELPER — for testing retrieval on its own
# ─────────────────────────────────────────────────────────────

def display_retrieved_chunks(query: str, chunks: list) -> None:
    """
    Pretty-prints retrieved chunks to the terminal for inspection.

    Args:
        query: The question that was searched for.
        chunks: The list of Document objects the retriever returned.
    """

    console.print(f"Query: {query}")
    console.print(f"Retrieved {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown source")

        page = chunk.metadata.get("page", 0)
        # FIX: previously defaulted to "?" here. If page metadata was
        # ever missing, "?" + 1 below would crash with a TypeError,
        # since you can't add an int to a string. Defaulting to 0
        # (matching the convention already used in query_expansion.py's
        # format_context()) keeps this safe and consistent across files.

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Key", style="bold yellow", width=10)
        tbl.add_column("value", style="white")
        tbl.add_row("Source", os.path.basename(source))
        tbl.add_row("Page", str(page + 1))
        # +1 because PyPDFLoader's "page" metadata is 0-indexed
        # internally, but humans expect page 1, not page 0.
        tbl.add_row("Chars", str(len(chunk.page_content)))

        console.print(Panel(
            f"{tbl}\n\n[dim]{chunk.page_content[:300]}...[/dim]",
            title=f"[bold white]Chunk {i}[/bold white]",
            border_style="cyan",
        ))


# ─────────────────────────────────────────────────────────────
# ENTRY POINT — lets you test retrieval on its own, without
# query expansion or the LLM answer step layered on top.
# ─────────────────────────────────────────────────────────────

def main():
    vector_store = load_vector_store(FAISS_INDEX_PATH)
    retriever = build_retriever(vector_store)

    test_query = input("Enter your query here: ")
    chunks = retriever.invoke(test_query)
    display_retrieved_chunks(test_query, chunks)


if __name__ == "__main__":
    main()