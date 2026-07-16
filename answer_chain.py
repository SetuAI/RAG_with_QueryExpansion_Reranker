##############################################################################
# answer_chain.py
#
# PURPOSE:
#   This is the final stage of the RAG pipeline, shared by any script
#   that needs to turn a pile of retrieved chunks into a written answer.
#   It formats chunks into a labelled context block, and builds the
#   prompt -> LLM -> parser chain that generates the final answer.
#
# WHY THIS FILE EXISTS — THE BOUNDARY PRINCIPLE:
#   This logic originally lived only inside query_expansion.py. But once
#   reranker.py also needed to show a final answer (not just a reranked
#   chunk list) in its own standalone demo, the same two functions would
#   have had to be copy-pasted into a second file — meaning any future
#   change to the answer prompt (tone, model, guardrails) would need to
#   be made in two places and could quietly drift apart. Pulling it out
#   into its own file means there is exactly one place that defines
#   "how we turn chunks into an answer," and both query_expansion.py and
#   reranker.py just call it.
#
# SHOULD YOU RUN THIS FILE DIRECTLY?
#   No — this file only defines functions. It gets imported by
#   query_expansion.py and reranker.py.
#
# HOW OTHER FILES USE THIS:
#   from answer_chain import format_context, build_answer_chain
#
#   context_str = format_context(chunks)
#   answer_chain = build_answer_chain()
#   answer = answer_chain.invoke({"context": context_str, "question": query})
##############################################################################


import os
from dotenv import load_dotenv
load_dotenv(override=True)
# Loads OPENAI_API_KEY, needed here since build_answer_chain() creates
# its own ChatOpenAI instance.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

ANSWER_MODEL = "gpt-4o"
# The final answer benefits from a stronger model, since it has to
# read a merged/reranked context accurately and stay precise with
# numbers — this is a financial-reporting assistant, so precision
# matters more here than in the query expansion step.

ANSWER_TEMPERATURE = 0
# Zero randomness for the final answer — we want the same question
# against the same context to produce a consistent, repeatable
# answer, not creative variation.


# ─────────────────────────────────────────────────────────────
# STEP 1: FORMAT CHUNKS FOR THE PROMPT
# ─────────────────────────────────────────────────────────────

def format_context(chunks: list) -> str:
    """
    Converts a list of chunks into a single labelled text block,
    ready to be inserted as {context} into the answer prompt.

    Args:
        chunks: The chunks to include as context — in our pipeline,
            this is the output of reranker.py's rerank(), or the raw
            merged chunks from query_expansion.py if reranking is
            skipped.

    Returns:
        A string with every chunk's text, each labelled with its
        source file and page number so the LLM (and a human reading
        the final answer) can trace claims back to a specific report.
    """

    if not chunks:
        # Guards against the answer prompt receiving an empty context
        # block, which would otherwise silently read as "CONTEXT:
        # " with nothing after it — confusing for both the LLM and
        # anyone debugging why an answer came back wrong.
        return "No context retrieved."

    formatted_parts = []
    for chunk in chunks:
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", 0) + 1
        formatted_parts.append(
            f"[SOURCE: {source} | PAGE: {page}]\n{chunk.page_content}\n"
        )

    return "\n\n".join(formatted_parts)


# ─────────────────────────────────────────────────────────────
# STEP 2: BUILD THE ANSWER CHAIN
# ─────────────────────────────────────────────────────────────

def build_answer_chain(model: str = ANSWER_MODEL, temperature: float = ANSWER_TEMPERATURE):
    """
    Builds the final answer-generation chain: prompt -> LLM -> parser.

    Args:
        model: Which chat model to use for the final answer.
        temperature: Sampling temperature for the answer LLM.

    Returns:
        A LangChain runnable chain. Calling .invoke({"context": ...,
        "question": ...}) on it returns the final answer as a string.
    """

    answer_llm = ChatOpenAI(model=model, temperature=temperature)

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a financial analyst assistant for Arctis Technologies Ltd.
            Your job is to answer questions about quarterly financial performance.

            Answer ONLY from the context provided below. Be precise with numbers, percentages,
            and quarter references. If the answer is not found in the context, say clearly:
            "I could not find that information in the available quarterly reports."

            Do not speculate. Do not use information outside the provided context.

            CONTEXT:
            {context}""",
        ),
        ("human", "{question}"),
    ])

    output_parser = StrOutputParser()
    return answer_prompt | answer_llm | output_parser