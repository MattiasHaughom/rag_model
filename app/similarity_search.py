from .database.vector_store import VectorStore
from .services.synthesizer import Synthesizer
from timescale_vector import client

class SearchService:
    def __init__(self):
        self.vec = VectorStore()
    
    def perform_search(self, query):
        reranked_results = self.vec.hybrid_search(
            query=query, keyword_k=10, semantic_k=10, rerank=True, top_n=5
        )
        response = Synthesizer.generate_response(question=query, context=reranked_results)
        
        # Return the structured response directly instead of trying to access 'answer'
        return response