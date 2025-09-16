import numpy as np
import os
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib

class LightweightEmbeddings:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.global_vocabulary = set()
        self.is_fitted = False
        self.document_embeddings_cache = {}
    
    def _get_fixed_vectorizer(self, texts: List[str]) -> TfidfVectorizer:
        """Create or get a fixed TF-IDF vectorizer with consistent vocabulary"""
        # Update global vocabulary with new texts
        for text in texts:
            words = text.lower().split()
            self.global_vocabulary.update(words)
        
        # Create vectorizer with fixed vocabulary if not exists
        if self.tfidf_vectorizer is None:
            # Start with a reasonable vocabulary size
            vocab_list = sorted(list(self.global_vocabulary))[:1000]  # Limit to 1000 most common words
            
            self.tfidf_vectorizer = TfidfVectorizer(
                vocabulary=vocab_list,
                stop_words='english',
                ngram_range=(1, 1),  # Only unigrams for consistency
                lowercase=True,
                max_features=1000
            )
            self.is_fitted = True
        
        return self.tfidf_vectorizer
    
    def get_embeddings_tfidf(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using TF-IDF with consistent dimensions"""
        try:
            if not texts:
                return []
            
            # Get or create vectorizer
            vectorizer = self._get_fixed_vectorizer(texts)
            
            # Transform texts to TF-IDF vectors
            tfidf_matrix = vectorizer.transform(texts)
            embeddings = tfidf_matrix.toarray().tolist()
            
            # Ensure consistent dimensions
            target_dim = 1000  # Fixed dimension
            normalized_embeddings = []
            
            for embedding in embeddings:
                if len(embedding) < target_dim:
                    # Pad with zeros
                    embedding.extend([0.0] * (target_dim - len(embedding)))
                elif len(embedding) > target_dim:
                    # Truncate
                    embedding = embedding[:target_dim]
                normalized_embeddings.append(embedding)
            
            return normalized_embeddings
            
        except Exception as e:
            print(f"TF-IDF embedding error: {e}")
            # Fallback to simple word embeddings
            return self._simple_word_embeddings(texts)
    
    def _simple_word_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback: Simple word-based embeddings with fixed dimensions"""
        # Create a fixed vocabulary from all texts
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        # Use top 1000 words for consistency
        word_list = sorted(list(all_words))[:1000]
        
        # Pad to exactly 1000 dimensions
        while len(word_list) < 1000:
            word_list.append(f"pad_{len(word_list)}")
        
        embeddings = []
        for text in texts:
            words = set(text.lower().split())
            embedding = [1.0 if word in words else 0.0 for word in word_list]
            embeddings.append(embedding)
        
        return embeddings
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query with consistent dimensions"""
        try:
            if self.tfidf_vectorizer is None:
                # If no vectorizer exists, create simple embedding
                return self._simple_word_embeddings([query])[0]
            
            # Transform query using existing vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            embedding = query_vector.toarray()[0].tolist()
            
            # Ensure consistent dimensions
            target_dim = 1000
            if len(embedding) < target_dim:
                embedding.extend([0.0] * (target_dim - len(embedding)))
            elif len(embedding) > target_dim:
                embedding = embedding[:target_dim]
            
            return embedding
            
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return self._simple_word_embeddings([query])[0]
    
    def find_relevant_chunks(self, query: str, document_chunks: List[str], 
                           document_embeddings: List[List[float]], top_k: int = 3) -> List[dict]:
        """Find most relevant chunks using cosine similarity with robust error handling"""
        try:
            if not document_chunks or not document_embeddings:
                return []
            
            # Get query embedding
            query_embedding = self.get_query_embedding(query)
            
            # Ensure all embeddings have the same dimension
            target_dim = 1000
            
            # Normalize query embedding
            if len(query_embedding) != target_dim:
                if len(query_embedding) < target_dim:
                    query_embedding.extend([0.0] * (target_dim - len(query_embedding)))
                else:
                    query_embedding = query_embedding[:target_dim]
            
            # Normalize document embeddings
            normalized_doc_embeddings = []
            for doc_emb in document_embeddings:
                if len(doc_emb) != target_dim:
                    if len(doc_emb) < target_dim:
                        doc_emb.extend([0.0] * (target_dim - len(doc_emb)))
                    else:
                        doc_emb = doc_emb[:target_dim]
                normalized_doc_embeddings.append(doc_emb)
            
            # Calculate cosine similarities
            similarities = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            for doc_emb in normalized_doc_embeddings:
                doc_emb_np = np.array(doc_emb).reshape(1, -1)
                
                # Handle zero vectors
                if np.all(query_embedding_np == 0) or np.all(doc_emb_np == 0):
                    similarity = 0.0
                else:
                    similarity = cosine_similarity(query_embedding_np, doc_emb_np)[0][0]
                
                similarities.append(similarity)
            
            # Get top k most similar chunks
            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(document_chunks) and similarities[idx] > 0.1:  # Threshold for relevance
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
        try:
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
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []

# Global embeddings instance
embeddings_engine = LightweightEmbeddings()