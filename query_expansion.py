##############################################################################
# query_expansion.py
#
# PURPOSE:
#   This is stage 3, the main demo entry point, of the query-expansion
#   RAG pipeline. Given a user's question, it:
#
#     PART A — QUERY EXPANSION (advanced retrieval technique)
#       1. Asks an LLM to generate several alternative phrasings of
#          the original question.
#       2. Runs a separate FAISS search for the original question AND
#          for every alternative phrasing.
#       3. Merges all the results together and removes duplicate chunks.
#
#     PART B — RERANKING (precision stage, see reranker.py)
#       4. Re-scores the merged chunks against the ORIGINAL question
#          using a cross-encoder, and keeps only the top few by
#          genuine relevance — narrowing the wide net expansion cast
#          down to what's actually worth showing the LLM.
#
#     PART C — ANSWER CHAIN
#       5. Feeds the reranked chunks into a prompt template along
#          with the original question.
#       6. Sends that to gpt-4o, which writes the final answer.
#
# WHY QUERY EXPANSION HELPS:
#   A single search only finds chunks that are close, in meaning, to
#   the EXACT wording the user typed. If a report says "Net Profit
#   After Tax" and the user asks about "PAT," those are the same idea
#   phrased differently, and a single search might rank the right
#   chunk lower than it deserves — or miss it. By generating several
#   alternative phrasings and searching for each one, we widen the
#   net: any chunk that matches ANY reasonable phrasing of the
#   question gets a chance to be found.
#
# TRADE-OFFS (worth saying out loud in the demo):
#   - Cost: N embedding calls + N FAISS searches instead of 1, plus
#     one extra LLM call to generate the expansions in the first place.
#   - Latency: the expansion call alone typically adds ~200-400ms
#     before retrieval even starts.
#   - Token usage: merging results from multiple searches usually
#     means a longer final context, which means a more expensive
#     final LLM call too.
#   - Net result: meaningfully better recall, at a real cost in
#     latency and money. Worth it when missing the right chunk is
#     more expensive than a slower, pricier query.
#
# WHY THIS FILE EXISTS — THE BOUNDARY PRINCIPLE:
#   This file owns the "expand, search, merge, answer" workflow.
#   It does NOT own how the vector store is loaded or configured —
#   that responsibility belongs entirely to retriever.py, and this
#   file only ever calls get_retriever() to get one. That way, if we
#   change TOP_K or the embedding model, we change it in exactly one
#   place (retriever.py), and this file automatically follows along.
#
# TRACING WITH LANGSMITH:
#   Every chain in this file is built with LangChain's pipe syntax
#   (prompt | llm | parser), which means each .invoke() call is
#   automatically traced end-to-end if LangSmith is configured. No
#   code changes are needed here — just set these in your .env file:
#
#       LANGCHAIN_TRACING_V2=true
#       LANGCHAIN_API_KEY=<your LangSmith API key>
#       LANGCHAIN_PROJECT=<a project name to group these runs under>
#
#   With that set, every expansion call and every answer generation
#   shows up as its own traced run in your LangSmith project — useful
#   for showing participants exactly what prompt went in and what
#   came back, without needing extra print statements for the LLM
#   calls themselves (the terminal output below is for the RETRIEVAL
#   side, which LangSmith doesn't visualize as clearly).
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   Yes — this is the demo entry point. ingest.py must have been run
#   at least once first, since this file depends on the FAISS index
#   it produces (via retriever.py).
##############################################################################


import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# WORKAROUND: on macOS, faiss and numpy/torch can each bundle their
# own copy of the OpenMP runtime (libomp.dylib). When both load in
# the same process, OpenMP aborts with "Error #15: already
# initialized" instead of silently picking one. This must be set
# before faiss gets imported anywhere — including indirectly, via
# the `from retriever import get_retriever` line below — so it goes
# at the very top of this file, before every other import.

from dotenv import load_dotenv
load_dotenv(override=True)
# Loads OPENAI_API_KEY, and if present, the LANGCHAIN_* variables
# described above for tracing.

from langchain_openai import ChatOpenAI
# Wraps OpenAI's chat models. Used here twice, with two different
# models: a cheap/fast one for generating query phrasings, and a
# stronger one for writing the final answer.

from langchain_core.prompts import ChatPromptTemplate
# Builds reusable prompt templates with {placeholder} slots that get
# filled in at invoke() time.

from langchain_core.output_parsers import StrOutputParser
# Converts the LLM's raw AIMessage response object into a plain
# Python string, so the rest of our code doesn't have to unwrap it.

from rich.console import Console
from rich.panel import Panel
# rich formatting for readable terminal output — this is what makes
# the query expansion process visible step-by-step during a live demo.

from retriever import get_retriever
# The single source of truth for retrieval configuration. See the
# "WHY THIS FILE EXISTS" note above — we deliberately do NOT rebuild
# a retriever from scratch here.

from reranker import rerank
# Optional precision stage. Expansion (above) optimizes for recall —
# cast a wide net across phrasings so we don't miss the right chunk.
# Reranking optimizes for precision — take everything expansion found
# and cut it down to the genuinely most relevant few, in real relevance
# order, using a cross-encoder instead of plain vector similarity.
# See reranker.py's header for the full explanation of why this is a
# separate, later step rather than something folded into retrieval.

from answer_chain import format_context, build_answer_chain
# The shared final-answer stage — see answer_chain.py's header for why
# this moved into its own file. reranker.py's standalone demo mode
# uses this same import to show a final answer too, without duplicating
# the prompt and chain-building logic in two places.

console = Console()


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

EXPANSION_MODEL = "gpt-4o-mini"
# Cheap and fast is fine here — generating alternative phrasings of
# a question doesn't need a top-tier model, and this call happens
# on every single query, so cost adds up fast if it's not cheap.

EXPANSION_TEMPERATURE = 0.4
# A bit of randomness is desirable here — we WANT some variation in
# phrasing across the generated alternatives, rather than the model
# giving four near-identical rewordings.

NUM_EXPANSIONS = 5
# How many alternative phrasings to generate, in addition to the
# original query. With the original included, this means up to
# 6 separate FAISS searches run per user question.

CHUNK_PREVIEW_CHARS = 220
# How much of each retrieved chunk's text to show in the terminal
# per-query breakdown. Long enough to recognize the content, short
# enough to keep the demo output scannable.

RERANK_TOP_N = 5
# How many chunks survive the reranking step and actually make it into
# the final answer prompt. Deliberately smaller than the merged chunk
# count from expansion — reranking's job is to cut the wide net down
# to the genuinely best few, not just reorder everything.


# ─────────────────────────────────────────────────────────────
# PART A, STEP 1: BUILD THE QUERY EXPANSION CHAIN
# ─────────────────────────────────────────────────────────────

def build_query_expander(
    model: str = EXPANSION_MODEL,
    temperature: float = EXPANSION_TEMPERATURE,
):
    """
    Builds and returns the query expansion chain: prompt -> LLM -> parser.

    Args:
        model: Which chat model to use for generating phrasings.
        temperature: Sampling temperature for the expansion LLM.

    Returns:
        A LangChain runnable chain. Calling .invoke({"question": ...,
        "num_expansions": ...}) on it returns a plain string containing
        one alternative phrasing per line.
    """

    expansion_llm = ChatOpenAI(model=model, temperature=temperature)

    expansion_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert at reformulating financial queries for document retrieval.
        Given the user's original question below, generate {num_expansions} alternative
        phrasings that mean the same thing, but use different vocabulary.

        Focus on financial terminology, different time references, and varied sentence
        structures to maximize retrieval coverage.

        Original Question: {question}
        Output ONLY the {num_expansions} alternative phrasings, one per line.
        DO NOT number them. DO NOT add explanation or any preamble.
        """
    )

    raw_parser = StrOutputParser()

    return expansion_prompt | expansion_llm | raw_parser


def expand_query(original_query: str, num_expansions: int = NUM_EXPANSIONS) -> list:
    """
    Generates alternative phrasings of the user's question.

    Args:
        original_query: The question exactly as the user typed it.
        num_expansions: How many alternative phrasings to generate.
            FIX: previously this parameter was accepted but silently
            ignored — the function body always used the NUM_EXPANSIONS
            global instead, so calling expand_query(q, num_expansions=3)
            had no effect. It's now threaded through to both the prompt
            call and the expander build, so the parameter actually does
            what it says.

    Returns:
        A list starting with the original query, followed by
        num_expansions alternative phrasings.
    """

    console.print(f"\n[bold]Running query expansion...[/bold]")
    console.print(f"Original query: {original_query}")

    expansion_chain = build_query_expander()
    raw_output = expansion_chain.invoke({
        "question": original_query,
        "num_expansions": num_expansions,
    })

    # The LLM returns one phrasing per line as plain text. We split on
    # newlines and drop any blank lines (e.g. a trailing newline at
    # the end of the response).
    expanded_queries = [
        line.strip()
        for line in raw_output.strip().split("\n")
        if line.strip()
    ]

    # Always include the original query itself in the search set —
    # expansion should ADD coverage, never replace the user's actual
    # question.
    all_queries = [original_query] + expanded_queries

    console.print(f"Expanded into {len(all_queries)} total queries:")
    for i, q in enumerate(all_queries):
        label = "[bold white](original)[/bold white]" if i == 0 else f"[yellow](expansion {i})[/yellow]"
        console.print(f"  {label} {q}")

    return all_queries


# ─────────────────────────────────────────────────────────────
# PART A, STEP 2: RUN RETRIEVAL FOR EVERY PHRASING AND MERGE
# ─────────────────────────────────────────────────────────────

def retrieve_with_expansion(original_query: str, retriever) -> list:
    """
    Runs the full expand -> retrieve -> merge -> dedupe workflow, and
    prints a per-query breakdown to the terminal so the whole process
    is visible during a live demo.

    Args:
        original_query: The question exactly as the user typed it.
        retriever: A retriever from get_retriever() (retriever.py).

    Returns:
        A deduplicated list of chunks (Document objects), merged
        across every query phrasing's search results.
    """

    all_queries = expand_query(original_query)

    seen_content = set()   # tracks chunk text we've already kept, for deduplication
    merged_chunks = []     # the running list of unique chunks across all searches

    console.print(f"\n[bold]Running {len(all_queries)} retrieval searches...[/bold]")

    for i, query in enumerate(all_queries):
        chunks = retriever.invoke(query)
        new_count = 0

        label = "(original)" if i == 0 else f"(expansion {i})"
        console.print(f"\n[cyan]Query {i + 1} {label}:[/cyan] \"{query}\"")

        # Single pass per chunk: decide new-vs-duplicate AND print the
        # preview line at the same time, so the "NEW"/"duplicate" marker
        # is always based on the exact same check that drives merging —
        # no risk of the display logic and the dedup logic disagreeing.
        for chunk in chunks:
            is_new = chunk.page_content not in seen_content
            if is_new:
                seen_content.add(chunk.page_content)
                merged_chunks.append(chunk)
                new_count += 1

            source = os.path.basename(chunk.metadata.get("source", "unknown"))
            page = chunk.metadata.get("page", 0) + 1
            marker = "[green]NEW[/green]" if is_new else "[dim]duplicate[/dim]"
            preview = chunk.page_content[:CHUNK_PREVIEW_CHARS].replace("\n", " ")
            console.print(f"    [{marker}] {source} (p.{page}): {preview}...")

        console.print(
            f"  -> retrieved {len(chunks)} chunk(s), {new_count} new unique"
        )
        # Note: participants will also notice the SAME chunk showing up
        # as "duplicate" under a later query phrasing — that overlap is
        # worth pointing out live, since it's a sign different phrasings
        # are converging on the right answer.

    console.print(
        f"\n[bold]Merged result:[/bold] {len(merged_chunks)} unique chunk(s) "
        f"after deduplication across {len(all_queries)} queries\n"
    )

    return merged_chunks


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    retriever = get_retriever()
    # Loaded from retriever.py — see the "WHY THIS FILE EXISTS" note
    # at the top of this file for why we don't build one here directly.

    answer_chain = build_answer_chain()

    console.print("\n[bold]Type your question. Type 'exit' to quit.[/bold]")

    while True:
        query = input("\nEnter the query: ").strip()

        if query.lower() in ("exit", "quit", "q"):
            console.print("Exiting...")
            break

        if not query:
            console.print("Please enter a query.")
            continue

        merged_chunks = retrieve_with_expansion(query, retriever)
        reranked_chunks = rerank(query, merged_chunks, top_n=RERANK_TOP_N)
        context_str = format_context(reranked_chunks)

        console.print("[bold]Generating answer with gpt-4o...[/bold]")
        answer = answer_chain.invoke({
            "context": context_str,
            "question": query,
        })

        console.print(Panel(
            answer,
            title="[bold green]Final Answer[/bold green]",
            border_style="green",
        ))


if __name__ == "__main__":
    main()