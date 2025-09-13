import google.generativeai as genai
import numpy as np
import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

class LightweightEmbeddings:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features for efficiency
            stop_words='english',
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        self.is_fitted = False
    
    def get_embeddings_tfidf(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using TF-IDF (lightweight alternative)"""
        try:
            if not self.is_fitted:
                # Fit the vectorizer on the texts
                self.tfidf_vectorizer.fit(texts)
                self.is_fitted = True
            
            # Transform texts to TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            return tfidf_matrix.toarray().tolist()
        except Exception as e:
            print(f"TF-IDF embedding error: {e}")
            # Fallback to simple word count vectors
            return self._simple_word_embeddings(texts)
    
    def _simple_word_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback: Simple word-based embeddings"""
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        word_list = sorted(list(all_words))[:500]  # Limit to 500 most common words
        
        embeddings = []
        for text in texts:
            words = text.lower().split()
            embedding = [1.0 if word in words else 0.0 for word in word_list]
            embeddings.append(embedding)
        
        return embeddings
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        if self.is_fitted:
            query_vector = self.tfidf_vectorizer.transform([query])
            return query_vector.toarray()[0].tolist()
        else:
            # If not fitted, use simple approach
            return self._simple_word_embeddings([query])[0]
    
    def find_relevant_chunks(self, query: str, document_chunks: List[str], 
                           document_embeddings: List[List[float]], top_k: int = 3) -> List[dict]:
        """Find most relevant chunks using cosine similarity"""
        try:
            query_embedding = self.get_query_embedding(query)
            
            # Calculate cosine similarities
            similarities = []
            for doc_emb in document_embeddings:
                similarity = cosine_similarity([query_embedding], [doc_emb])[0][0]
                similarities.append(similarity)
            
            # Get top k most similar chunks
            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Lower threshold for TF-IDF
                    results.append({
                        'chunk_index': int(idx),
                        'content': document_chunks[idx],
                        'relevance_score': float(similarities[idx])
                    })
            
            return results
        except Exception as e:
            print(f"Error in relevance search: {e}")
            # Fallback to simple keyword matching
            return self._simple_keyword_search(query, document_chunks, top_k)
    
    def _simple_keyword_search(self, query: str, chunks: List[str], top_k: int = 3) -> List[dict]:
        """Fallback: Simple keyword-based search"""
        query_words = set(query.lower().split())
        
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            score = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
            chunk_scores.append((i, score))
        
        # Sort by score and take top k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in chunk_scores[:top_k]:
            if score > 0:
                results.append({
                    'chunk_index': idx,
                    'content': chunks[idx],
                    'relevance_score': score
                })
        
        return results

# Global embeddings instance
embeddings_engine = LightweightEmbeddings()