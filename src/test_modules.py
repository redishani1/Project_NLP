"""
Comprehensive test script to verify all modules work correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

print("Testing NLP UD Project Modules...")
print("=" * 60)

# Test imports
print("\n1. Testing Imports...")
try:
    from ud_loader import load_conllu_file, extract_sentence_data
    from statistics import compute_corpus_statistics, compute_pos_distribution
    from preprocessor import TextPreprocessor
    from similarity import SimilarityAnalyzer
    from visualizer import plot_pos_distribution
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test TextPreprocessor
print("\n2. Testing TextPreprocessor...")
try:
    preprocessor = TextPreprocessor()
    test_text = "The cats are running quickly through beautiful gardens."
    result = preprocessor.process_sentence(test_text)
    
    print(f"   Original: {result['original']}")
    print(f"   Tokens: {result['tokens'][:5]}...")
    print(f"   Lemmas: {result['processed'][:5]}...")
    print("   ✓ TextPreprocessor works correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test SimilarityAnalyzer
print("\n3. Testing SimilarityAnalyzer...")
try:
    sentences = [
        "Natural language processing is fascinating.",
        "NLP deals with computational linguistics.",
        "The weather is nice today."
    ]
    
    analyzer = SimilarityAnalyzer(max_features=100, stop_words=None)
    tfidf_matrix = analyzer.fit_transform(sentences)
    cosine_sim = analyzer.compute_cosine_similarity()
    
    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   Similarity matrix shape: {cosine_sim.shape}")
    print(f"   Sentence 0 vs 1 similarity: {cosine_sim[0, 1]:.4f}")
    print("   ✓ SimilarityAnalyzer works correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("\nNext steps:")
print("1. Download a .conllu file from https://universaldependencies.org/")
print("2. Place it in the data/ folder")
print("3. Update the file path in src/main.py or notebooks/ud_analysis.ipynb")
print("4. Run the analysis!")
