##############################################################################
# ingest.py
#
# PURPOSE:
#   This is stage 1 of the query-expansion RAG pipeline.
#   It reads the raw quarterly PDF reports, cuts them into small
#   overlapping text chunks, converts every chunk into a numeric
#   vector (an embedding), and saves the whole thing to disk as a
#   FAISS index.
#
# WHY THIS FILE EXISTS — THE BOUNDARY PRINCIPLE:
#   Think of this file as the loading dock.
#   Raw PDFs come in. A searchable vector index goes out.
#   Everything downstream — retriever.py, query_expansion.py — never
#   touches a PDF directly. They only ever talk to the FAISS index
#   this file produces. If we ever swap PyPDFLoader for a different
#   PDF parser, or change how we chunk, this is the only file that
#   needs to change.
#
# WHY WE CHUNK BEFORE EMBEDDING:
#   Embedding models have a token limit, and a whole 3-page PDF report
#   won't fit inside one embedding meaningfully anyway — a single vector
#   for an entire document would blur together the revenue numbers, the
#   CEO commentary, and the balance sheet into one vague point in space.
#   Small chunks let each vector represent one specific idea, so a
#   search for "Q2 EBITDA" can match the exact paragraph that mentions it.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   Yes — this is the entry point for stage 1. Run it once whenever
#   you add, remove, or change the source PDFs. It must be run before
#   retriever.py or query_expansion.py, since both depend on the FAISS
#   index this file creates.
#
# HOW OTHER FILES USE THIS:
#   This file is not imported anywhere — it only writes files to disk.
#   retriever.py and query_expansion.py load its output like this:
#
#       from langchain_community.vectorstores import FAISS
#       vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, ...)
##############################################################################


import os
# os lets us list files in a directory and build file paths safely
# across operating systems.

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# WORKAROUND: on macOS, faiss and numpy/torch can each bundle their
# own copy of the OpenMP runtime (libomp.dylib). When both get loaded
# in the same process, OpenMP aborts with "Error #15: already
# initialized" instead of silently picking one. This env var tells
# OpenMP to tolerate the duplicate rather than crash. It must be set
# BEFORE faiss is imported below, which is why this line comes first.
# setdefault() means it won't override a value you've already set
# yourself in the shell or .env file.

from dotenv import load_dotenv
load_dotenv(override=True)
# Loads OPENAI_API_KEY (and any other secrets) from a local .env file
# into the environment, so OpenAIEmbeddings can authenticate below.
# override=True means values in .env always win over any values
# already set in the shell environment.

from langchain_community.document_loaders import PyPDFLoader
# Reads a PDF file page by page and returns one LangChain Document
# object per page, with the page text plus metadata (source path,
# page number) attached automatically.

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Cuts long text into smaller overlapping chunks. "Recursive" means
# it tries the most natural break point first (paragraph), and only
# falls back to a rougher break (single character) if it has to.

from langchain_openai import OpenAIEmbeddings
# Wraps OpenAI's embedding API. Turns a chunk of text into a list of
# numbers (a vector) that captures its meaning.

from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search) is the vector database.
# It stores every chunk's vector and lets us search for the closest
# ones to a query vector, extremely fast, even across thousands of
# chunks.

from rich.console import Console
from rich.panel import Panel
# rich just gives us nicer, colour-coded terminal output so we can
# see ingestion progress clearly. Purely cosmetic — has no effect
# on the actual pipeline logic.

console = Console()


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
#
# Every tunable setting lives here, at the top, in one place.
# Nothing below this block should ever hard-code a value that
# belongs here instead.
# ─────────────────────────────────────────────────────────────

DOCUMENTS_DIR = "."
# Folder to scan for source PDFs. "." means "the same folder this
# script is run from." Place the quarterly report PDFs there before
# running this file.

FAISS_INDEX_PATH = "./faiss_index"
# Where the finished vector index gets saved. retriever.py and
# query_expansion.py both read from this exact path, so if you
# change it here, update it in both of those files too.

CHUNK_SIZE = 700
# Target size of each chunk, in characters. Small enough that each
# chunk stays focused on one idea (e.g. one table or one paragraph
# of commentary), large enough to keep useful context together.

CHUNK_OVERLAP = 250
# How many characters consecutive chunks share. Without overlap, a
# sentence that happens to fall right on a chunk boundary gets cut
# in half, and neither resulting chunk makes full sense on its own.
# Overlap protects against that at the cost of some duplicate text
# across the index.

EMBEDDING_MODEL = "text-embedding-3-large"
# Must be the exact same embedding model used later in retriever.py
# and query_expansion.py. Vectors from different embedding models
# live in different, incompatible number-spaces — mixing them would
# silently produce meaningless similarity scores.


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD THE RAW PDFS
# ─────────────────────────────────────────────────────────────

def load_documents(documents_dir: str) -> list:
    """
    Scans documents_dir for PDF files and loads every page of every
    PDF into a LangChain Document object.

    Args:
        documents_dir: Folder to scan for files ending in .pdf.

    Returns:
        A list of Document objects, one per PDF page, each carrying
        the page's text plus metadata (source file path, page number).

    Raises:
        FileNotFoundError: If no .pdf files are found in documents_dir.
        We raise here rather than silently continuing with zero
        documents, because an empty index would fail in a confusing
        way much later, at query time, instead of right here.
    """

    pdf_files = [
        os.path.join(documents_dir, f)
        for f in os.listdir(documents_dir)
        if f.lower().endswith(".pdf")
    ]
    # .lower() makes the check case-insensitive, so "Report.PDF" is
    # still picked up, not just "report.pdf".

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{documents_dir}'. "
            "Please place the report PDFs in that folder before running ingest.py."
        )

    console.print(f"Found {len(pdf_files)} PDF file(s)")

    all_documents = []
    for pdf_path in sorted(pdf_files):
        # sorted() keeps ingestion order predictable and repeatable
        # (e.g. Q1 before Q2 before Q3) regardless of how the
        # operating system happens to list files.
        console.print(f"Loading: {os.path.basename(pdf_path)}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Each page becomes its own Document. A 4-page PDF produces
        # 4 Document objects here, each with metadata={"source": ..., "page": ...}.

        all_documents.extend(pages)

    console.print(f"Total pages loaded: {len(all_documents)}")
    return all_documents


# ─────────────────────────────────────────────────────────────
# STEP 2: SPLIT PAGES INTO SMALLER CHUNKS
# ─────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Splits full-page Document objects into smaller overlapping chunks
    sized for embedding.

    Args:
        documents: List of page-level Document objects from load_documents().

    Returns:
        A list of smaller Document objects (chunks), each inheriting
        the source and page metadata of the page it came from.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        # Tried in order: split on blank lines (paragraphs) first,
        # then single line breaks, then sentence ends, then spaces,
        # and only as a last resort cut mid-word. This keeps chunks
        # as semantically coherent as possible.
    )

    chunks = splitter.split_documents(documents)
    console.print(f"Total chunks created: {len(chunks)}")

    if chunks:
        # Print a preview of the first chunk so you can sanity-check
        # that chunking looks reasonable before waiting for embeddings.
        console.print(Panel(
            chunks[0].page_content[:400] + "...",
            title="Sample chunk (first 400 characters)",
            border_style="dim",
        ))

    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 3: EMBED CHUNKS AND BUILD THE VECTOR STORE
# ─────────────────────────────────────────────────────────────

def embed_and_store(chunks: list) -> FAISS:
    """
    Converts every chunk's text into a vector and builds a searchable
    FAISS index from all of them.

    Args:
        chunks: List of chunk-level Document objects from split_documents().

    Returns:
        A FAISS vector store containing one vector per chunk, ready
        to be searched or saved to disk.
    """

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    # This is the expensive step — it makes one embedding API call
    # per chunk (LangChain batches this internally) and builds the
    # FAISS index from the results.

    console.print(f"Total vectors in index: {vector_store.index.ntotal}")
    return vector_store


# ─────────────────────────────────────────────────────────────
# STEP 4: SAVE THE VECTOR STORE TO DISK
# ─────────────────────────────────────────────────────────────

def save_vector_store(vector_store: FAISS, path: str) -> None:
    """
    Persists the FAISS index to disk so later scripts (retriever.py,
    query_expansion.py) can load it without re-embedding anything.

    Args:
        vector_store: The FAISS store built by embed_and_store().
        path: Folder to save into. Created if it doesn't exist.
    """

    vector_store.save_local(path)
    console.print(f"Vector store saved to {path}")
    console.print("Files created: index.faiss, index.pkl")
    # index.faiss  -> the raw vectors, in FAISS's binary format
    # index.pkl    -> the chunk text + metadata (pickled Python objects),
    #                 needed to map a matched vector back to its source text


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    documents = load_documents(DOCUMENTS_DIR)
    chunks = split_documents(documents)
    vector_store = embed_and_store(chunks)
    save_vector_store(vector_store, FAISS_INDEX_PATH)


if __name__ == "__main__":
    main()