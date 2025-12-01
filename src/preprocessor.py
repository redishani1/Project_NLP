"""
Module for text preprocessing, tokenization, and lemmatization.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import Dict, List


# Download required NLTK data (run once)
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


class TextPreprocessor:
    """
    Text preprocessing class for tokenization, lemmatization, and stemming.
    """
    
    def __init__(self):
        """Initialize preprocessor with lemmatizer and stemmer."""
        download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def process_sentence(self, text: str, use_stemming: bool = False, 
                        lowercase: bool = True) -> Dict[str, any]:
        """
        Process a sentence with tokenization and lemmatization/stemming.
        
        Args:
            text: Input sentence string
            use_stemming: If True, use stemming; otherwise use lemmatization
            lowercase: Convert to lowercase
            
        Returns:
            Dictionary with processed information:
                - original: Original text
                - tokens: List of tokens
                - pos_tags: List of (token, tag) tuples
                - processed: Lemmatized or stemmed tokens
                - method: Processing method used
        """
        # Optionally convert to lowercase
        process_text = text.lower() if lowercase else text
        
        # Tokenization
        tokens = word_tokenize(process_text)
        
        # PoS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Lemmatization or Stemming
        if use_stemming:
            processed = [self.stemmer.stem(token) for token in tokens]
            method = "Stemming"
        else:
            processed = [self.lemmatizer.lemmatize(token) for token in tokens]
            method = "Lemmatization"
        
        return {
            'original': text,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'processed': processed,
            'method': method
        }
    
    def batch_process(self, texts: List[str], use_stemming: bool = False) -> List[Dict]:
        """
        Process multiple sentences.
        
        Args:
            texts: List of sentence strings
            use_stemming: If True, use stemming; otherwise use lemmatization
            
        Returns:
            List of processed results
        """
        return [self.process_sentence(text, use_stemming) for text in texts]
    
    def get_lemmas(self, text: str) -> List[str]:
        """
        Get lemmatized tokens from text.
        
        Args:
            text: Input text
            
        Returns:
            List of lemmas
        """
        result = self.process_sentence(text, use_stemming=False)
        return result['processed']
    
    def get_stems(self, text: str) -> List[str]:
        """
        Get stemmed tokens from text.
        
        Args:
            text: Input text
            
        Returns:
            List of stems
        """
        result = self.process_sentence(text, use_stemming=True)
        return result['processed']
