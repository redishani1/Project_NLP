"""
Module for TF-IDF vectorization and similarity analysis.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple, Dict


class SimilarityAnalyzer:
    """
    Class for TF-IDF vectorization and similarity calculations.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 2), 
                 stop_words: str = 'english', min_df: int = 2):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to extract
            stop_words: Stop words to use
            min_df: Minimum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df
        )
        self.tfidf_matrix = None
        self.feature_names = None
    
    def fit_transform(self, sentences: List[str]):
        """
        Fit vectorizer and transform sentences to TF-IDF matrix.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            TF-IDF matrix (sparse)
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(sentences)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix
    
    def get_sentence_tfidf(self, sentence_idx: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get top TF-IDF weighted terms for a specific sentence.
        
        Args:
            sentence_idx: Index of the sentence
            top_n: Number of top terms to return
            
        Returns:
            DataFrame with terms and their TF-IDF scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        sentence_vector = self.tfidf_matrix[sentence_idx]
        
        # Get non-zero features and their scores
        feature_indices = sentence_vector.nonzero()[1]
        tfidf_scores = [(self.feature_names[i], sentence_vector[0, i]) 
                       for i in feature_indices]
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        
        df = pd.DataFrame(tfidf_scores[:top_n], columns=['Term', 'TF-IDF Score'])
        df['TF-IDF Score'] = df['TF-IDF Score'].round(4)
        
        return df
    
    def compute_cosine_similarity(self) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Returns:
            Cosine similarity matrix
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        return cosine_similarity(self.tfidf_matrix)
    
    def compute_euclidean_distance(self) -> np.ndarray:
        """
        Compute pairwise Euclidean distance matrix.
        
        Returns:
            Euclidean distance matrix
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        return euclidean_distances(self.tfidf_matrix)
    
    def find_most_similar_pairs(self, sentences: List[str], 
                               similarity_matrix: np.ndarray, 
                               top_n: int = 10) -> List[Tuple]:
        """
        Find the most similar sentence pairs.
        
        Args:
            sentences: List of sentence texts
            similarity_matrix: Cosine similarity matrix
            top_n: Number of top pairs to return
            
        Returns:
            List of tuples (idx1, idx2, similarity_score, sent1, sent2)
        """
        pairs = []
        n = similarity_matrix.shape[0]
        
        # Only consider upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, similarity_matrix[i, j], 
                            sentences[i], sentences[j]))
        
        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_n]
    
    def compare_sentences(self, sent1_idx: int, sent2_idx: int,
                         cosine_matrix: np.ndarray,
                         euclidean_matrix: np.ndarray) -> Dict:
        """
        Compare two sentences with multiple metrics.
        
        Args:
            sent1_idx: Index of first sentence
            sent2_idx: Index of second sentence
            cosine_matrix: Cosine similarity matrix
            euclidean_matrix: Euclidean distance matrix
            
        Returns:
            Dictionary with comparison metrics
        """
        return {
            'sentence_1_idx': sent1_idx,
            'sentence_2_idx': sent2_idx,
            'cosine_similarity': round(cosine_matrix[sent1_idx, sent2_idx], 4),
            'euclidean_distance': round(euclidean_matrix[sent1_idx, sent2_idx], 4)
        }
    
    def get_similarity_statistics(self, similarity_matrix: np.ndarray) -> Dict:
        """
        Get statistics about the similarity matrix.
        
        Args:
            similarity_matrix: Similarity matrix
            
        Returns:
            Dictionary with statistics
        """
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu_indices(similarity_matrix.shape[0], k=1)
        values = similarity_matrix[upper_triangle]
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q1': float(np.percentile(values, 25)),
            'q3': float(np.percentile(values, 75))
        }
