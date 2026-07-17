
### What is a Reranker

**A Reranker is a model that takes a list of documents and reorders them, putting the most relevant ones at the top for a given question.**

In simple words, a Reranker is a smart sorter. We give it a question and a small pile of documents. It reads each document carefully against the question and decides which one answers the question best. Then it arranges them from most relevant to least relevant.

Let's say we go to a library and ask the librarian for books on "healthy cooking". A junior helper quickly runs around and brings us 50 books that have the words "healthy" or "cooking" somewhere on the cover. That is a fast but rough first pass.

Now, a senior librarian sits down, opens each of those 50 books, and actually reads a few pages. She knows that one book is about cooking for athletes, another is a diet plan, and another only mentions cooking once in passing. She reorders the pile so the best matches are on top.

That senior librarian is the Reranker.

So, the Reranker does not search the whole world. It only re-arranges a shortlist that someone else already gathered. It does not invent new documents and it does not go and find more. It simply reorders what it is given. This single idea is the heart of everything we will learn today.


---

Question
   |
   v
[ Retriever ]   --> fetches many candidate documents (fast, rough)
   |
   v
[ Reranker ]    --> reorders the candidates by true relevance (slow, precise)
   |
   v
[ LLM ]         --> reads the best documents and writes the final answer
   |
   v
Answer

---

---

**We can see that the Reranker sits right in the middle. It comes after the first fetch and before the AI writes the answer. Its only job is to clean up and reorder what the first step found, so that the AI reads the best material.**



### The two-stage retrieval idea

Imagine we have a huge collection of documents. It can be millions of web pages, or thousands of company documents, or the chapters of many books. We call this big collection the** ** **corpus** .

When a user asks a question, in theory the perfect approach would be to carefully read every single document in the corpus and pick the best ones. But reading millions of documents carefully for every question would take forever. It would be too slow and too expensive.

So, here comes the two-stage approach to the rescue.

**Stage 1 - Retrieval:** A fast retriever quickly scans the entire corpus and pulls out a list of maybe 100 or 200 candidate documents that look roughly related to the question. This stage is built for speed. It is allowed to be a little sloppy.

**Stage 2 - Reranking:** The Reranker takes only that shortlist of 100 or 200 candidates and reads each one carefully against the question. It then reorders them and we keep only the top few, for example the top 5 or top 10.

Let's see this with a simple picture:

```
Corpus (1,000,000 documents)
        |
        |  Stage 1: fast retriever
        v
   100 candidates  (rough shortlist)
        |
        |  Stage 2: reranker
        v
   Top 5 documents  (carefully reordered, most relevant)
```

Here, we can notice the trick. We never run the slow, careful model on all one million documents. We only run it on the 100 that survived the first stage. This way we get both speed and accuracy.

This is the central idea. The first stage trades precision for speed. The second stage trades speed for precision. Together, they give us the best of both.

### Why first-stage retrieval is fast but not precise

Now, let's understand the first stage better, because understanding its weakness is exactly why the Reranker exists.

There are two common ways to do first-stage retrieval, and many real systems combine both into what is called** **[Hybrid Search](https://outcomeschool.com/blog/how-does-hybrid-search-work). Let's understand both in simple words.

**BM25 (keyword search):** This is the classic way. It matches the words in the question with the words in the documents. If the question says "healthy cooking", it looks for documents that contain those exact words and rewards documents where the rarer, more meaningful words appear more often. It is fast and it works well, but it only understands exact words. It does not understand meaning. If a document says "nutritious recipes" instead of "healthy cooking", BM25 will struggle, because the words are different even though the meaning is the same.

**Vector search (semantic search):** This is the modern way. Here, every document is converted into a list of numbers called an** ** **embedding** . An embedding is a way to represent the meaning of a piece of text as a point in space. Texts with similar meaning end up close to each other, and texts with different meaning end up far apart. When a question comes, we turn the question into an embedding too, and then we look for the documents whose embeddings are closest to the question's embedding.

We have a detailed blog on** **[how a Vector Database works](https://outcomeschool.com/blog/how-does-a-vector-database-work) that explains embeddings and vector search in depth.

Vector search is wonderful because it understands meaning, not just words. But it is still not perfect.

Here is the catch. In vector search, the question and the documents are turned into embeddings completely separately, ahead of time. The model never gets to look at the question and a document together. It compresses an entire document into one fixed list of numbers before it ever sees our question. A lot of fine detail is lost in that compression.

Let's say we ask, "Which medicine should I avoid if I have high blood pressure?" A document that talks a lot about blood pressure and medicines in general will look very close in vector space, even if it never actually says which medicine to avoid. The embedding captures the broad topic, but it can miss the precise answer.

So, first-stage retrieval is fast and gives us a good rough shortlist. But it often puts a "kind of related" document above the "exactly right" document. It gets us into the right neighborhood, but not to the right house.

One way to improve the first stage itself is** **[HyDE](https://outcomeschool.com/blog/how-does-hyde-work), where we search with a hypothetical answer instead of the raw question, so the query is shaped more like the documents we want.

This is exactly the gap the Reranker fills. To understand how it does a better job, we must learn about two types of models.

### Bi-encoder vs Cross-encoder

This is the most important part of the blog, so let's go slowly.

A first-stage retriever using vector search is built on a** ** **bi-encoder** . The Reranker is built on a** ** **cross-encoder** . The names sound similar, but the difference between them is the whole secret.

**Bi-encoder:** The word "bi" means two. A bi-encoder uses two separate passes. It encodes the question on its own, and it encodes each document on its own, into separate embeddings. The question never meets the document inside the model. Only at the very end do we compare the two embeddings to see how close they are.

Let's picture it like below:

```
BI-ENCODER (used for fast retrieval)

  Question  -->  [encoder]  -->  [ vector A ]
                                       \
                                        compare distance --> score
                                       /
  Document  -->  [encoder]  -->  [ vector B ]
```

Here, the big advantage is speed. Because each document is encoded by itself, we can encode all our documents once, in advance, and store their embeddings. When a question arrives, we only encode the question and compare. That is why vector search over millions of documents can be very fast. The structure that makes this fast over so many vectors is** **[Approximate Nearest Neighbor (ANN) search](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work), which skips most of the comparisons.

But the disadvantage is precision. The model decided what each document "means" without ever knowing what we would ask. It had to guess in advance which details matter.

**Cross-encoder:** The word "cross" means crossing or combining. A cross-encoder takes the question and one document and feeds them in together, joined as a single input. The model reads them side by side and can directly relate each word of the question to each word of the document. At the end, it outputs a single number, a** ** **relevance score** , that says how well this document answers this question.

Let's picture it like below:

```
CROSS-ENCODER (used for reranking)

  Question + Document  -->  [ model reads them together ]  -->  relevance score (for example, 0.92)
```

Here, the advantage is accuracy. Because the model sees the question and the document at the same time, it can catch fine details. It can notice that the document really does name the exact medicine to avoid, and not just talk about medicines in general.

But the disadvantage is speed. The cross-encoder must run fresh for every single question-document pair. Nothing can be prepared in advance, because the question is new every time. If we had a million documents, we would have to run the cross-encoder a million times for one question. That is far too slow.

Let me tabulate the differences between a bi-encoder and a cross-encoder for your better understanding so that you can decide which one to use based on your use case.

| Point           | Bi-encoder                                 | Cross-encoder                  |
| --------------- | ------------------------------------------ | ------------------------------ |
| How it reads    | Question and document separately           | Question and document together |
| Output          | One embedding per text                     | One relevance score per pair   |
| Speed           | Very fast                                  | Slow                           |
| Precision       | Good, but loses fine detail                | High, catches fine detail      |
| Can precompute? | Yes, embed documents in advance            | No, must run per question      |
| Best used for   | First-stage retrieval over the full corpus | Reranking a small shortlist    |

So now we can clearly see the plan. We use the fast bi-encoder to shortlist many candidates, and then we use the accurate cross-encoder to reorder only that shortlist. The Reranker is the cross-encoder.



### How a Reranker scores documents step by step

Now, let's walk through exactly what a Reranker does when a question comes in. We will use small numbers for the sake of understanding.

Suppose the user asks: "What is the capital of France?"

The first-stage retriever returns 4 candidate documents:

* Document 1: "France is a country in Europe known for its food and culture."
* Document 2: "Paris is the capital and largest city of France."
* Document 3: "The Eiffel Tower is located in Paris and is very famous."
* Document 4: "Berlin is the capital of Germany."

**Step 1:** The Reranker pairs the question with each document, one at a time. So we form 4 pairs.

**Step 2:** It feeds each pair into the cross-encoder, which reads the question and that document together and produces a relevance score. A higher score means more relevant.

Let's say the scores come out like below:

```
Question: "What is the capital of France?"

(Question + Document 1)  -->  cross-encoder  -->  0.34
(Question + Document 2)  -->  cross-encoder  -->  0.97
(Question + Document 3)  -->  cross-encoder  -->  0.55
(Question + Document 4)  -->  cross-encoder  -->  0.21
```

Here, we can see that Document 2 scored the highest, because it directly says Paris is the capital of France. Document 3 scored in the middle because it mentions Paris but does not actually answer the question. Document 4 scored low because it is about Germany, even though it shares the word "capital". The exact numbers do not matter on their own. What matters is the order they give us.

**Step 3:** The Reranker sorts the documents by score, from highest to lowest.

After sorting, the order becomes: Document 2, Document 3, Document 1, Document 4.

We can picture this reordering as below:

```
   BEFORE (first-stage order)            AFTER (reranked order)

   +-----------------------+             +-----------------------+
   | 1. Document 1  (0.34) |             | 1. Document 2  (0.97) |
   | 2. Document 4  (0.21) |   rerank    | 2. Document 3  (0.55) |
   | 3. Document 2  (0.97) |   ------->  | 3. Document 1  (0.34) |
   | 4. Document 3  (0.55) |             | 4. Document 4  (0.21) |
   +-----------------------+             +-----------------------+
```

Here, we can notice that the first-stage order on the left is messy, because Document 4 about Germany was placed high just for sharing the word "capital", while the truly correct Document 2 was buried lower. After reranking on the right, the documents are sorted purely by their relevance score, so Document 2 rises to the very top and Document 4 sinks to the bottom.

**Step 4:** We keep only the top-k documents. If we choose top-2, we keep Document 2 and Document 3.

Notice something important. The first-stage retriever often places a document like Document 4 high, because it shares the strong word "capital" with the question. The Reranker fixed this mistake by actually understanding the meaning. It pushed the truly correct document, Document 2, to the very top.

This is how a Reranker works in practice. It scores, it sorts, and we take the top few. Problem Solved.

In a RAG system, those top-2 documents are then handed to the AI model, which reads them and writes a clean answer like "The capital of France is Paris." The better the documents we hand over, the better the final answer. The Reranker makes sure we hand over the best documents.



### The accuracy vs latency and cost trade-off

Now, we must talk about the cost, because this is exactly why we do not just rerank everything. Two things matter here. One is** ** **latency** , which simply means how long the user waits for an answer. The other is the money cost of running the model.

A cross-encoder is powerful but slow and expensive to run. Every question-document pair needs a full pass through the model. So the more documents we rerank, the more time and money it takes.

Let's say running the cross-encoder on one pair takes a tiny moment. Running it on 100 pairs is fine, it stays fast enough for a user to wait. But running it on one million pairs for every question would be painfully slow and would cost a fortune.

This is the whole reason for the two-stage design. We let the cheap, fast bi-encoder handle the millions, and we save the expensive, accurate cross-encoder for the small shortlist.

```
Run cross-encoder on ALL 1,000,000 docs  -->  extremely slow and costly  (we avoid this)

Run cross-encoder on TOP 100 only         -->  fast enough and very accurate  (we do this)
```

So, the Reranker gives us a careful, high-quality reordering, but we apply it only where it pays off. We balance accuracy against latency and cost by choosing how many candidates to rerank. A common choice is to retrieve 100 to 200 candidates and rerank them down to the top 5 or top 10.

**Note:** The number of candidates we rerank is a knob we can tune. More candidates means higher accuracy but more cost and a slower response. Fewer candidates means faster and cheaper but we risk missing a good document that the first stage ranked low. We pick the balance based on our use case.

### Late-interaction models like ColBERT

We have seen two extremes. The bi-encoder is fast but rough. The cross-encoder is precise but slow. Now, the next question is, can we have something in the middle? The answer is yes.

There is a family of models called** ** **late-interaction models** , and the most well-known one is called** ** **ColBERT** , which is short for** ** **Contextualized Late Interaction over BERT** .

In simple words, a late-interaction model is a clever middle ground between a bi-encoder and a cross-encoder.

Let's understand the idea gently. A normal bi-encoder squeezes a whole document into one single embedding. ColBERT does not do that. Instead, it keeps a separate small embedding for each word in the document, and it can compute these document embeddings in advance, just like a bi-encoder. That keeps it fast.

Then, at question time, it compares the words of the question against the words of the document in a fine-grained way and adds up the best matches. This word-by-word comparison happens late, which is why it is called "late interaction". It recovers a lot of the fine detail that a normal bi-encoder loses.

So, ColBERT sits between the two worlds. It is more precise than a plain bi-encoder because it compares at the word level, and it is faster than a cross-encoder because it precomputes the document side. We can think of it as a faster, lighter way to get some of the Reranker's accuracy.

Now, an important question is, where exactly does ColBERT fit in our pipeline? This matters, because ColBERT can sit in two different places. We can picture both placements as below:

```
Option A - ColBERT as the Reranker (Stage 2 only):

   Question --> [ Retriever ] --> [ ColBERT ] --> [ LLM ] --> Answer

Option B - ColBERT as Retriever and Reranker together (Stage 1 + Stage 2):

   Question --> [ ColBERT ] --> [ LLM ] --> Answer
```

Here, we can see that in Option A, ColBERT replaces only the Reranker box, while a normal fast retriever still does the rough first-stage fetch. In Option B, because ColBERT precomputes a small embedding for every word in every document, it can do the first-stage retrieval itself, so it collapses the Retriever and the Reranker into a single step. This is special to ColBERT. A plain cross-encoder can never do Option B, because it must read the question and the document together, so it can only work on a small shortlist that some other retriever has already prepared.

So, how do we choose between the two? Each one has its own advantage over the other.

**Advantage of Option A:** It is easy to add to a system we already have, because we keep our existing fast retriever and simply let ColBERT rerank the short shortlist it produces. If we encode that shortlist on the fly, we can even keep the index small.

**Advantage of Option B:** It gives better recall, because ColBERT searches the whole corpus at the word level itself, so a weak first stage cannot drop a good document before it reaches reranking. The cost is a much larger index, because ColBERT must store a separate embedding for every word in every document of the whole corpus, not just for a small shortlist.

There is one more detail worth knowing, which is how ColBERT handles the document side.

**Note:** There are two ways to handle the document side when we use ColBERT. The common way is to pre-encode every document once, store these word-level embeddings, and at question time encode only the question and compare it against the stored embeddings. This is the standard ColBERT approach. It gives very low latency, but it needs a larger index. The other way is to skip storing them and encode the shortlisted documents on the fly at question time, which keeps storage low but makes each question slower. We can summarize the trade-off as below:

| Approach                        | Storage | Query latency |
| ------------------------------- | ------- | ------------- |
| Pre-encode and store (standard) | High    | Low           |
| Encode on the fly               | Low     | High          |

Here, we can see that pre-encoding is the usual choice for large systems, because storing the embeddings once lets every later question run fast.



### Real examples of Rerankers

Now that we understand the idea, let's see some real Rerankers that people use in production today.

**BGE reranker:** BGE is a family of open-source models. The BGE reranker is a cross-encoder we can download and run on our own machines. Because it is open-source and free to use, it is very popular for teams who want full control and want to avoid sending their data to an outside service.

**ColBERT:** As we just learned, ColBERT is the well-known late-interaction model. It is used both as a retriever and as a reranker, and it is a good option when we want strong accuracy with better speed than a full cross-encoder.

So, whether we choose a hosted reranking service or a self-hosted option such as a BGE reranker, the core working is the same. The model reads the question together with each candidate document and produces a relevance score, and we reorder by that score.

### Why Rerankers matter for RAG

Let's close by tying everything back to the big picture.

In a RAG system, the AI model can only answer well if we feed it good documents. If the first-stage retriever puts mediocre documents at the top, the AI reads mediocre material and writes a weak or even wrong answer. This is one of the most common reasons RAG answers disappoint people.

The Reranker fixes exactly this. It takes the rough shortlist and makes sure the genuinely relevant documents rise to the top before they ever reach the AI. When we feed in better documents, we get back a better answer.

Let's say a company has a help system built on RAG. Without a Reranker, a user asks "How do I get a refund?" and the system feeds the AI a document about shipping, because it happened to share some words. The answer comes out confused. With a Reranker, the refund policy document is correctly pushed to the top, and the AI gives a clear, correct answer.

This is the quiet power of a Reranker. It is a small, focused step, but it lifts the quality of the entire system. It makes our life easier.

So, to recap the full journey. A fast first-stage retriever uses a bi-encoder to pull a rough shortlist from a huge corpus. A Reranker, built on a cross-encoder, reads the question together with each shortlisted document, scores the relevance, and reorders them so the best ones land on top. We run this slow, careful step only on the shortlist to balance accuracy against latency and cost, and we can reach for a middle-ground model like ColBERT when we want speed and precision together.iislelff




---
