"""
Module for computing corpus statistics.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Any


def compute_corpus_statistics(corpus_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics about the corpus.
    
    Args:
        corpus_data: Dictionary with extracted corpus data
        
    Returns:
        Dictionary with statistics
    """
    num_sentences = len(corpus_data['sentence_texts'])
    num_tokens = len(corpus_data['all_tokens'])
    vocabulary_size = len(set(corpus_data['all_lemmas']))
    unique_pos_tags = len(set(corpus_data['all_pos_tags']))
    
    # Calculate sentence length statistics
    sent_lengths = [len(sent) for sent in corpus_data['sentence_tokens']]
    avg_sent_length = np.mean(sent_lengths)
    median_sent_length = np.median(sent_lengths)
    std_sent_length = np.std(sent_lengths)
    
    # Type-Token Ratio
    ttr = vocabulary_size / num_tokens if num_tokens > 0 else 0
    
    return {
        'num_sentences': num_sentences,
        'num_tokens': num_tokens,
        'vocabulary_size': vocabulary_size,
        'unique_pos_tags': unique_pos_tags,
        'avg_sent_length': avg_sent_length,
        'median_sent_length': median_sent_length,
        'std_sent_length': std_sent_length,
        'min_sent_length': min(sent_lengths) if sent_lengths else 0,
        'max_sent_length': max(sent_lengths) if sent_lengths else 0,
        'type_token_ratio': ttr
    }


def compute_pos_distribution(pos_tags: List[str]) -> pd.DataFrame:
    """
    Calculate frequency distribution of PoS tags.
    
    Args:
        pos_tags: List of all PoS tags
        
    Returns:
        DataFrame with PoS tag frequencies and percentages
    """
    pos_counter = Counter(pos_tags)
    pos_df = pd.DataFrame(pos_counter.most_common(), columns=['PoS Tag', 'Frequency'])
    pos_df['Percentage'] = (pos_df['Frequency'] / pos_df['Frequency'].sum() * 100).round(2)
    
    return pos_df


def get_top_frequent_words(tokens: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Get the most frequent words in the corpus.
    
    Args:
        tokens: List of all tokens
        top_n: Number of top words to return
        
    Returns:
        DataFrame with top frequent words
    """
    word_counter = Counter(tokens)
    freq_df = pd.DataFrame(word_counter.most_common(top_n), 
                          columns=['Word', 'Frequency'])
    freq_df['Rank'] = range(1, len(freq_df) + 1)
    
    return freq_df[['Rank', 'Word', 'Frequency']]


def get_top_frequent_lemmas(lemmas: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Get the most frequent lemmas in the corpus.
    
    Args:
        lemmas: List of all lemmas
        top_n: Number of top lemmas to return
        
    Returns:
        DataFrame with top frequent lemmas
    """
    lemma_counter = Counter(lemmas)
    freq_df = pd.DataFrame(lemma_counter.most_common(top_n), 
                          columns=['Lemma', 'Frequency'])
    freq_df['Rank'] = range(1, len(freq_df) + 1)
    
    return freq_df[['Rank', 'Lemma', 'Frequency']]


def create_statistics_summary(stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a formatted DataFrame with statistics summary.
    
    Args:
        stats: Dictionary with statistics
        
    Returns:
        DataFrame with formatted statistics
    """
    summary_data = [
        {'Metric': 'Total Sentences', 'Value': f"{stats['num_sentences']:,}"},
        {'Metric': 'Total Tokens', 'Value': f"{stats['num_tokens']:,}"},
        {'Metric': 'Vocabulary Size (Unique Lemmas)', 'Value': f"{stats['vocabulary_size']:,}"},
        {'Metric': 'Unique PoS Tags', 'Value': f"{stats['unique_pos_tags']}"},
        {'Metric': 'Average Sentence Length', 'Value': f"{stats['avg_sent_length']:.2f}"},
        {'Metric': 'Median Sentence Length', 'Value': f"{stats['median_sent_length']:.2f}"},
        {'Metric': 'Std Dev Sentence Length', 'Value': f"{stats['std_sent_length']:.2f}"},
        {'Metric': 'Min Sentence Length', 'Value': f"{stats['min_sent_length']}"},
        {'Metric': 'Max Sentence Length', 'Value': f"{stats['max_sent_length']}"},
        {'Metric': 'Type-Token Ratio', 'Value': f"{stats['type_token_ratio']:.4f}"},
    ]
    
    return pd.DataFrame(summary_data)
