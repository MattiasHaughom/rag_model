from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client


# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

#irrelevant_question = "What is the weather in Tokyo?"

#results = vec.search(irrelevant_question, limit=3)

#response = Synthesizer.generate_response(question=irrelevant_question, context=results)

#print(f"\n{response.answer}")
#print("\nThought process:")
#for thought in response.thought_process:
#    print(f"- {thought}")
#print(f"\nContext: {response.enough_context}")

query = "Who are the leadership team at kommunalbanken?"

# add more advanced filtering methods to get the main keywords from the questions

# --------------------------------------------------------------
# Semantic search
# --------------------------------------------------------------

semantic_results = vec.semantic_search(query=query, limit=5)


# --------------------------------------------------------------
# Simple keyword search
# --------------------------------------------------------------

keyword_results = vec.keyword_search(query=query, limit=5)


# --------------------------------------------------------------
# Hybrid search
# --------------------------------------------------------------

hybrid_results = vec.hybrid_search(query=query, keyword_k=10, semantic_k=10)


# --------------------------------------------------------------
# Reranking
# --------------------------------------------------------------

reranked_results = vec.hybrid_search(
    query=query, keyword_k=10, semantic_k=10, rerank=True, top_n=5
)

# --------------------------------------------------------------
# Synthesize
# --------------------------------------------------------------

response = Synthesizer.generate_response(question=query, context=reranked_results)
print(response.answer)