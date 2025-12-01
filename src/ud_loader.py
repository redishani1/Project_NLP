"""
Module for loading and parsing Universal Dependencies .conllu files.
"""

from conllu import parse, parse_incr
from typing import List, Dict, Any
from pathlib import Path


def load_conllu_file(file_path: str) -> List:
    """
    Load and parse a .conllu file.
    
    Args:
        file_path: Path to the .conllu file
        
    Returns:
        List of parsed sentences
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = parse(f.read())
        return sentences
    except Exception as e:
        raise ValueError(f"Error parsing .conllu file: {e}")


def extract_sentence_data(sentences: List) -> Dict[str, Any]:
    """
    Extract tokens, lemmas, and PoS tags from parsed sentences.
    
    Args:
        sentences: List of parsed sentences from conllu
        
    Returns:
        Dictionary containing:
            - all_tokens: List of all tokens
            - all_lemmas: List of all lemmas
            - all_pos_tags: List of all PoS tags
            - sentence_texts: List of sentence texts
            - sentence_tokens: List of token lists per sentence
            - sentence_lemmas: List of lemma lists per sentence
    """
    all_tokens = []
    all_lemmas = []
    all_pos_tags = []
    sentence_texts = []
    sentence_tokens = []
    sentence_lemmas = []
    
    for sent in sentences:
        sent_tokens = []
        sent_lemmas = []
        
        for token in sent:
            # Skip multiword tokens (e.g., 1-2)
            if isinstance(token['id'], int):
                form = token['form']
                lemma = token['lemma']
                pos = token['upos']
                
                all_tokens.append(form)
                all_lemmas.append(lemma)
                all_pos_tags.append(pos)
                
                sent_tokens.append(form)
                sent_lemmas.append(lemma)
        
        # Get sentence text from metadata or reconstruct
        sent_text = sent.metadata.get('text', ' '.join(sent_tokens))
        sentence_texts.append(sent_text)
        sentence_tokens.append(sent_tokens)
        sentence_lemmas.append(sent_lemmas)
    
    return {
        'all_tokens': all_tokens,
        'all_lemmas': all_lemmas,
        'all_pos_tags': all_pos_tags,
        'sentence_texts': sentence_texts,
        'sentence_tokens': sentence_tokens,
        'sentence_lemmas': sentence_lemmas
    }


def get_conllu_sample(sentences: List, num_sentences: int = 5) -> List[Dict]:
    """
    Get a sample of sentences with their annotations.
    
    Args:
        sentences: List of parsed sentences
        num_sentences: Number of sentences to sample
        
    Returns:
        List of dictionaries with sentence information
    """
    sample_data = []
    
    for sent in sentences[:num_sentences]:
        sent_info = {
            'text': sent.metadata.get('text', ''),
            'tokens': [],
            'lemmas': [],
            'pos_tags': [],
            'features': []
        }
        
        for token in sent:
            if isinstance(token['id'], int):
                sent_info['tokens'].append(token['form'])
                sent_info['lemmas'].append(token['lemma'])
                sent_info['pos_tags'].append(token['upos'])
                sent_info['features'].append(token.get('feats', {}))
        
        sample_data.append(sent_info)
    
    return sample_data
