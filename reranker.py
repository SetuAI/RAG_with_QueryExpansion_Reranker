##############################################################################
# reranker.py
#
# PURPOSE:
#   This is an optional stage that sits between retrieval and the final
#   answer. It takes a pile of candidate chunks — in our pipeline, the
#   merged, deduplicated output of query_expansion.py's
#   retrieve_with_expansion() — and re-scores every one of them against
#   the ORIGINAL question, using a model that reads the query and the
#   chunk TOGETHER rather than comparing pre-computed vectors.
#
# WHY WE NEED THIS ON TOP OF QUERY EXPANSION:
#   Query expansion's job is RECALL — cast a wide net across several
#   phrasings so we don't miss the right chunk. But a wide net catches
#   some noise along with the good stuff, and the order those chunks
#   come back in is just "which search happened to find it first" —
#   not "how relevant is this to the actual question." Reranking's job
#   is PRECISION — take everything expansion found, and figure out
#   which pieces are actually worth putting in front of the LLM, in
#   genuine relevance order.
#
#   Think of it as: expansion widens the net, reranking tightens it.
#
# WHY A CROSS-ENCODER IS MORE ACCURATE THAN VECTOR SEARCH HERE:
#   FAISS similarity search (what retriever.py does) is a BI-ENCODER
#   approach: the query gets embedded into a vector, each chunk was
#   ALREADY embedded into a vector ahead of time, and we compare the
#   two vectors with cosine similarity. Fast, and works at huge scale —
#   but the query and the chunk were never actually read together.
#
#   A cross-encoder (what THIS file does) feeds the query and a chunk
#   into the model TOGETHER, as one input, and the model outputs a
#   single relevance score for that specific pair. This is much more
#   accurate, because the model can directly compare the two texts
#   word-for-word — but it's also much slower, since you have to run
#   the model once per (query, chunk) pair. That's exactly why we don't
#   use a cross-encoder for the initial search across an entire index —
#   it doesn't scale. We use it here, at the end, on a small shortlist
#   (a few dozen chunks) where the extra accuracy is worth the extra time.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   Yes — running it directly lets you test reranking in isolation: it
#   retrieves candidates for a query using the plain retriever (no
#   expansion), shows you the BEFORE order (plain similarity) next to
#   the AFTER order (cross-encoder relevance) with scores, and then
#   generates the final answer from the reranked chunks — the same
#   answer stage query_expansion.py uses, imported from answer_chain.py.
#   This is the clearest way to show a training audience the complete
#   retrieve -> rerank -> answer path on its own, before it's folded
#   into the full expansion pipeline.
#
# HOW OTHER FILES USE THIS:
#   from reranker import rerank
#
#   reranked_chunks = rerank(query, merged_chunks, top_n=5)
##############################################################################


import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Same OpenMP workaround as the other files — see the note in ingest.py.
# sentence-transformers pulls in torch, which can collide with faiss the
# same way, so this needs to be set before anything downstream imports it.

from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# A small, widely-used cross-encoder trained specifically for search
# relevance ranking (on the MS MARCO passage ranking dataset). It's
# ~80MB, runs fine on CPU, and downloads automatically the first time
# it's used — no API key required.

TOP_N = 5
# How many chunks to keep AFTER reranking. This is usually smaller than
# the number of candidate chunks going in — reranking's whole purpose
# is narrowing a bigger pile down to the genuinely best few, not just
# reordering the same number of chunks.


# ─────────────────────────────────────────────────────────────
# MODEL LOADING (lazy singleton)
# ─────────────────────────────────────────────────────────────

_reranker_model = None
# Module-level cache. The model is loaded from disk (or downloaded, on
# first use) exactly once per program run, not once per query — loading
# it fresh on every question would make the interactive demo loop in
# query_expansion.py noticeably slower after the very first question.

def get_reranker_model():
    """
    Loads the cross-encoder model, caching it after the first call.

    Returns:
        A sentence_transformers.CrossEncoder instance, ready to score
        (query, chunk) pairs.
    """

    global _reranker_model

    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        # Imported here, not at the top of the file, so that simply
        # importing reranker.py (e.g. to read TOP_N) doesn't force the
        # sentence-transformers library and its model weights to load
        # if reranking is never actually used.

        console.print(
            f"Loading reranker model '{RERANKER_MODEL}' "
            "(first use only, may take a moment)..."
        )
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        console.print("Reranker model loaded.")

    return _reranker_model


# ─────────────────────────────────────────────────────────────
# CORE RERANKING FUNCTION
# ─────────────────────────────────────────────────────────────

def rerank(query: str, chunks: list, top_n: int = TOP_N, show_comparison: bool = True) -> list:
    """
    Re-scores chunks against the query with a cross-encoder, and returns
    the top_n most relevant ones, in relevance order.

    Args:
        query: The user's original question (not an expanded phrasing —
            always rerank against what the user actually asked).
        chunks: Candidate chunks to score, in any order. In our pipeline
            this is the merged, deduplicated output of
            retrieve_with_expansion().
        top_n: How many chunks to keep after reranking.
        show_comparison: If True, prints a before/after table to the
            terminal so the reranking step is visible during a demo.

    Returns:
        The top_n chunks, sorted by relevance score, highest first.
        If chunks is empty, returns an empty list without loading the
        model at all.
    """

    if not chunks:
        return []

    model = get_reranker_model()

    # The cross-encoder wants a list of [query, passage] pairs — one
    # pair per chunk, all scored in a single batched call for speed.
    pairs = [[query, chunk.page_content] for chunk in chunks]
    scores = model.predict(pairs)

    # Pair each chunk with its score and its ORIGINAL position, then
    # sort by score, highest relevance first. Keeping the original
    # position lets us show how much reranking actually moved things
    # around, not just what the final order is.
    scored = [
        {"chunk": chunk, "score": float(score), "original_rank": i + 1}
        for i, (chunk, score) in enumerate(zip(chunks, scores))
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)

    if show_comparison:
        _display_before_after(query, chunks, scored, top_n)

    return [item["chunk"] for item in scored[:top_n]]


# ─────────────────────────────────────────────────────────────
# DISPLAY HELPER — makes the reranking step visible in the terminal
# ─────────────────────────────────────────────────────────────

def _display_before_after(query: str, original_chunks: list, scored: list, top_n: int) -> None:
    """
    Prints the candidate chunks as two clearly separate blocks — BEFORE
    reranking (original discovery order, no relevance score exists yet)
    and AFTER reranking (sorted by the cross-encoder's relevance score,
    with a rank-movement arrow showing how far each chunk climbed or
    fell). Two separate blocks, rather than one combined table, so the
    contrast is unmistakable to someone watching a live demo rather than
    something they have to read two columns to notice.

    Args:
        query: The question chunks were scored against, for display only.
        original_chunks: The chunks exactly as they were passed into
            rerank(), before any scoring — this is the "BEFORE" order.
        scored: The list of {"chunk", "score", "original_rank"} dicts
            produced by rerank(), already sorted by score — this is
            the "AFTER" order.
        top_n: How many of the top entries get marked as kept.
    """

    console.print(f"\n[bold]Reranking {len(scored)} candidate chunk(s) against:[/bold] \"{query}\"")

    # ── BEFORE ──────────────────────────────────────────────
    # This is just whatever order the chunks arrived in — retrieval
    # order, or merge/discovery order if this came from query expansion.
    # Notice there is deliberately no score shown here: nothing has
    # judged these chunks' relevance to THIS query yet.
    console.print("\n[bold]BEFORE reranking[/bold] [dim](original order — no relevance score yet)[/dim]")
    for i, chunk in enumerate(original_chunks, start=1):
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", 0) + 1
        preview = chunk.page_content[:100].replace("\n", " ")
        console.print(f"  #{i}  {source} (p.{page}): {preview}...")

    # ── AFTER ───────────────────────────────────────────────
    # Same chunks, now sorted purely by the cross-encoder's relevance
    # score. The movement arrow makes the reordering itself visible —
    # a chunk that jumped from #7 to #1 shows "^6", one that fell from
    # #1 to #5 shows "v4", and one that didn't move shows "-".
    console.print(f"\n[bold]AFTER reranking[/bold] [dim](sorted by relevance score, top {top_n} kept)[/dim]")
    for new_rank, item in enumerate(scored, start=1):
        source = os.path.basename(item["chunk"].metadata.get("source", "unknown"))
        page = item["chunk"].metadata.get("page", 0) + 1
        preview = item["chunk"].page_content[:100].replace("\n", " ")

        kept = new_rank <= top_n
        marker = "[green]KEPT[/green]" if kept else "[dim]dropped[/dim]"

        positions_moved = item["original_rank"] - new_rank
        if positions_moved > 0:
            movement = f"[green]^{positions_moved}[/green]"
        elif positions_moved < 0:
            movement = f"[red]v{abs(positions_moved)}[/red]"
        else:
            movement = "[dim]-[/dim]"

        console.print(
            f"  #{new_rank}  score={item['score']:.3f}  "
            f"(was #{item['original_rank']}, {movement})  {marker}  "
            f"{source} (p.{page}): {preview}..."
        )

    console.print(f"\nKept top {top_n} of {len(scored)} chunks after reranking.\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT — standalone before/after demo, no query expansion,
# no LLM answer step. Just plain retrieval vs. reranked retrieval,
# so the reranking step can be shown on its own.
# ─────────────────────────────────────────────────────────────

def main():
    from retriever import get_retriever
    # Imported here, not at the top of the file, to keep rerank()'s own
    # dependencies (just sentence-transformers) separate from this demo
    # entry point's dependency on the retrieval stage.

    from rich.panel import Panel
    from answer_chain import format_context, build_answer_chain
    # Also imported here rather than at the top of the file, for the
    # same reason as get_retriever above — rerank() itself never needs
    # these, only this standalone demo entry point does.

    retriever = get_retriever()
    answer_chain = build_answer_chain()

    test_query = input("\nEnter your query: ").strip()
    if not test_query:
        console.print("No query entered, exiting.")
        return

    console.print("\n[bold]Retrieving candidates (plain similarity search, no reranking)...[/bold]")
    chunks = retriever.invoke(test_query)
    console.print(f"Retrieved {len(chunks)} candidate chunk(s).")

    # rerank() prints its own BEFORE/AFTER comparison, so no extra
    # display code is needed here for that part.
    reranked_chunks = rerank(test_query, chunks, top_n=TOP_N)

    console.print("[bold]Generating answer with gpt-4o...[/bold]")
    context_str = format_context(reranked_chunks)
    answer = answer_chain.invoke({
        "context": context_str,
        "question": test_query,
    })

    console.print(Panel(
        answer,
        title="[bold green]Final Answer[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()