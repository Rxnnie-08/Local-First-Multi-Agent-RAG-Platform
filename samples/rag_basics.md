# Retrieval-Augmented Generation Basics

RAG gives a language model access to documents it was never trained on. Instead of fine-tuning, the
system retrieves relevant passages at question time and places them in the prompt.

## Chunking

Documents are split into chunks before embedding. Chunks that are too large dilute the meaning of
the embedding; chunks that are too small lose context. A common starting point is 300 to 500 tokens
with 10 to 15 percent overlap, so a sentence cut at a boundary still appears whole in one chunk.

## Dense versus sparse retrieval

Dense retrieval embeds the query and finds chunks with similar meaning, which handles paraphrase
well: "how do I make the model faster" can match a chunk about quantization. It is weak on exact
identifiers such as error codes, function names and part numbers.

Sparse retrieval scores exact term overlap in the BM25 family. It finds the chunk containing the
rare token immediately but misses paraphrases.

## Hybrid search

Hybrid search runs both retrievers and merges the rankings with Reciprocal Rank Fusion (RRF). A
chunk ranked highly by either retriever rises to the top, which makes hybrid more robust than
either alone.
