"""
Complete example script demonstrating the full UD analysis pipeline.
Run this script after placing a .conllu file in the data/ folder.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ud_loader import load_conllu_file, extract_sentence_data
from corpus_statistics import (compute_corpus_statistics, compute_pos_distribution,
                        create_statistics_summary, get_top_frequent_words,
                        get_top_frequent_lemmas)
from preprocessor import TextPreprocessor
from similarity import SimilarityAnalyzer
from visualizer import (plot_pos_distribution, plot_similarity_distribution,
                        plot_sentence_length_distribution, plot_top_frequent_words)
import numpy as np


def main():
    """Main analysis pipeline."""
    
    print("="*80)
    print("     NLP PROCESSING AND ANALYSIS WITH UNIVERSAL DEPENDENCIES")
    print("="*80)
    
    # Configuration
    CONLLU_FILE = "../data/sq_tsa-ud-test.conllu"  # Albanian TSA test corpus
    LANGUAGE = "Albanian"
    SUBSET_SIZE = 1000  # Number of sentences for TF-IDF analysis
    
    print(f"\n1. LOADING CORPUS")
    print(f"   Language: {LANGUAGE}")
    print(f"   File: {CONLLU_FILE}")
    
    try:
        sentences = load_conllu_file(CONLLU_FILE)
        print(f"   ✓ Loaded {len(sentences)} sentences")
    except FileNotFoundError:
        print(f"   ✗ Error: File not found: {CONLLU_FILE}")
        print(f"   Please download a .conllu file from https://universaldependencies.org/")
        return
    
    # Extract data
    print(f"\n2. EXTRACTING DATA")
    corpus_data = extract_sentence_data(sentences)
    print(f"   ✓ Extracted tokens, lemmas, and PoS tags")
    
    # Compute statistics
    print(f"\n3. COMPUTING STATISTICS")
    stats = compute_corpus_statistics(corpus_data)
    stats_df = create_statistics_summary(stats)
    print(f"\n{stats_df.to_string(index=False)}")
    
    # Save statistics
    stats_df.to_csv('../reports/corpus_statistics.csv', index=False)
    print(f"\n   ✓ Saved to reports/corpus_statistics.csv")
    
    # PoS distribution
    print(f"\n4. POS TAG DISTRIBUTION")
    pos_df = compute_pos_distribution(corpus_data['all_pos_tags'])
    print(f"\n   Top 10 PoS Tags:")
    print(f"   {pos_df.head(10).to_string(index=False)}")
    
    # Save PoS distribution
    pos_df.to_csv('../reports/pos_distribution.csv', index=False)
    print(f"\n   ✓ Saved to reports/pos_distribution.csv")
    
    # Visualize PoS distribution
    print(f"\n5. CREATING VISUALIZATIONS")
    plot_pos_distribution(pos_df, language=LANGUAGE, 
                         save_path='../outputs/pos_distribution.png')
    print(f"   ✓ PoS distribution plot saved")
    
    # Sentence length distribution
    sent_lengths = [len(sent) for sent in corpus_data['sentence_tokens']]
    plot_sentence_length_distribution(sent_lengths, language=LANGUAGE,
                                     save_path='../outputs/sentence_length_distribution.png')
    print(f"   ✓ Sentence length distribution plot saved")
    
    # Top frequent words
    print(f"\n6. FREQUENT WORDS ANALYSIS")
    top_words = get_top_frequent_words(corpus_data['all_tokens'], top_n=20)
    print(f"\n   Top 10 Words:")
    print(f"   {top_words.head(10).to_string(index=False)}")
    top_words.to_csv('../reports/top_frequent_words.csv', index=False)
    
    top_lemmas = get_top_frequent_lemmas(corpus_data['all_lemmas'], top_n=20)
    print(f"\n   Top 10 Lemmas:")
    print(f"   {top_lemmas.head(10).to_string(index=False)}")
    top_lemmas.to_csv('../reports/top_frequent_lemmas.csv', index=False)
    
    # Process new sentences
    print(f"\n7. PROCESSING NEW SENTENCES")
    preprocessor = TextPreprocessor()
    
    test_sentences = [
        "The cats are running quickly through the beautiful gardens.",
        "She has been studying natural language processing for three years."
    ]
    
    for i, sent in enumerate(test_sentences, 1):
        result = preprocessor.process_sentence(sent)
        print(f"\n   Example {i}:")
        print(f"   Original: {result['original']}")
        print(f"   Lemmas:   {result['processed'][:10]}...")
    
    # TF-IDF Analysis
    print(f"\n8. TF-IDF VECTORIZATION")
    subset_sentences = corpus_data['sentence_texts'][:SUBSET_SIZE]
    
    analyzer = SimilarityAnalyzer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None  # Albanian doesn't have built-in stop words in sklearn
    )
    
    tfidf_matrix = analyzer.fit_transform(subset_sentences)
    print(f"   ✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Sample TF-IDF scores
    sample_tfidf = analyzer.get_sentence_tfidf(5, top_n=5)
    print(f"\n   Sample TF-IDF scores (sentence 5):")
    print(f"   {sample_tfidf.to_string(index=False)}")
    
    # Similarity analysis
    print(f"\n9. SIMILARITY ANALYSIS")
    cosine_sim = analyzer.compute_cosine_similarity()
    euclidean_dist = analyzer.compute_euclidean_distance()
    print(f"   ✓ Computed similarity matrices")
    
    # Example comparisons
    print(f"\n   Example Comparisons:")
    comparison = analyzer.compare_sentences(10, 25, cosine_sim, euclidean_dist)
    print(f"   Sentence 10 vs 25:")
    print(f"   - Cosine Similarity: {comparison['cosine_similarity']}")
    print(f"   - Euclidean Distance: {comparison['euclidean_distance']}")
    
    # Statistics
    sim_stats = analyzer.get_similarity_statistics(cosine_sim)
    print(f"\n   Similarity Statistics:")
    print(f"   - Mean: {sim_stats['mean']:.4f}")
    print(f"   - Std Dev: {sim_stats['std']:.4f}")
    print(f"   - Min: {sim_stats['min']:.4f}")
    print(f"   - Max: {sim_stats['max']:.4f}")
    
    # Find most similar pairs
    print(f"\n10. MOST SIMILAR SENTENCE PAIRS")
    most_similar = analyzer.find_most_similar_pairs(subset_sentences, cosine_sim, top_n=5)
    
    for rank, (idx1, idx2, similarity, sent1, sent2) in enumerate(most_similar, 1):
        print(f"\n   Rank {rank} (Similarity: {similarity:.4f}):")
        print(f"   [{idx1}] {sent1[:80]}...")
        print(f"   [{idx2}] {sent2[:80]}...")
    
    # Save results
    import pandas as pd
    similar_results = []
    for idx1, idx2, similarity, sent1, sent2 in most_similar:
        similar_results.append({
            'Rank': len(similar_results) + 1,
            'Sentence_1_Index': idx1,
            'Sentence_2_Index': idx2,
            'Similarity': similarity,
            'Sentence_1': sent1[:100],
            'Sentence_2': sent2[:100]
        })
    
    similar_df = pd.DataFrame(similar_results)
    similar_df.to_csv('../reports/most_similar_pairs.csv', index=False)
    print(f"\n   ✓ Saved to reports/most_similar_pairs.csv")
    
    # Visualize similarity distribution
    print(f"\n11. SIMILARITY VISUALIZATION")
    upper_triangle = np.triu_indices(cosine_sim.shape[0], k=1)
    similarity_values = cosine_sim[upper_triangle]
    
    plot_similarity_distribution(similarity_values, 
                                save_path='../outputs/similarity_distribution.png')
    print(f"   ✓ Similarity distribution plot saved")
    
    print(f"\n" + "="*80)
    print(f"✓ ANALYSIS COMPLETE!")
    print(f"="*80)
    print(f"\nGenerated files:")
    print(f"  - reports/corpus_statistics.csv")
    print(f"  - reports/pos_distribution.csv")
    print(f"  - reports/top_frequent_words.csv")
    print(f"  - reports/top_frequent_lemmas.csv")
    print(f"  - reports/most_similar_pairs.csv")
    print(f"  - outputs/pos_distribution.png")
    print(f"  - outputs/sentence_length_distribution.png")
    print(f"  - outputs/similarity_distribution.png")
    print(f"\n")


if __name__ == "__main__":
    main()
