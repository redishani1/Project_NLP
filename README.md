# Albanian NLP Analysis with Universal Dependencies

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Project_NLP/blob/main/Albanian_NLP_Analysis.ipynb)

A comprehensive Python project for analyzing **Albanian language** linguistic data using the Universal Dependencies Albanian TSA treebank. This project demonstrates various NLP techniques including tokenization, lemmatization, PoS tagging, TF-IDF vectorization, and similarity analysis specifically for Albanian text.

> **🚀 Quick Start:** Click the "Open in Colab" badge above to run this project instantly in your browser - no setup required!

## 📋 Project Description

This project performs comprehensive analysis on the **Albanian (sq_tsa) Universal Dependencies corpus**, providing:
- Albanian corpus statistics and linguistic analysis
- Part-of-Speech tag distribution for Albanian
- Albanian text preprocessing (tokenization, lemmatization, stemming)
- TF-IDF vectorization for Albanian sentences
- Similarity analysis (cosine similarity, Euclidean distance)
- Visualization of Albanian linguistic patterns

## 🎯 Project Goals

Based on the course requirements, this project covers:
1. **Corpus Selection**: Work with Albanian UD treebank (sq_tsa)
2. **Data Loading**: Parse Albanian .conllu files and extract annotations
3. **Corpus Statistics**: Compute sentences, tokens, vocabulary size, PoS distributions for Albanian
4. **Preprocessing**: Implement tokenization, lemmatization, and stemming for Albanian text
5. **Vector Similarity**: Apply TF-IDF and compute cosine/Euclidean similarity on Albanian sentences
6. **Analysis**: Identify most similar Albanian sentence pairs

## 📁 Project Structure

```
Project_NLP/
│
├── data/                          # Albanian .conllu files
│   └── sq_tsa-ud-test.conllu     # Albanian TSA corpus
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── ud_loader.py              # Load and parse Albanian .conllu files
│   ├── corpus_statistics.py      # Albanian corpus statistics computation
│   ├── preprocessor.py           # Albanian text preprocessing
│   ├── similarity.py             # TF-IDF and similarity analysis
│   ├── visualizer.py             # Visualization functions
│   └── main.py                   # Complete example script
│
├── notebooks/                     # Jupyter notebooks
│   └── ud_analysis.ipynb         # Albanian analysis notebook
│
├── outputs/                       # Generated plots and visualizations
│
├── reports/                       # CSV reports and statistics
│   ├── corpus_statistics.csv
│   ├── pos_distribution.csv
│   ├── top_frequent_words.csv
│   ├── top_frequent_lemmas.csv
│   └── most_similar_pairs.csv
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Option 1: Google Colab (Easiest - No Setup Required!)

**Click the badge at the top of this README** or [open directly in Colab](https://colab.research.google.com/github/YOUR_USERNAME/Project_NLP/blob/main/Albanian_NLP_Analysis.ipynb)

The notebook will automatically:
- Clone the repository
- Install all dependencies
- Download the Albanian corpus
- Run the complete analysis

**Perfect for:** Quick demonstrations, sharing with others, presentations

### Option 2: Local Installation

**1. Install Dependencies**

```bash
pip install -r requirements.txt
```

**2. Albanian Dataset**

The project uses the **UD_Albanian-TSA** (Treebank of Spoken Albanian) corpus:
- **Test**: `sq_tsa-ud-test.conllu` - Albanian annotated corpus

The Albanian TSA treebank contains spontaneous spoken Albanian with rich linguistic annotations including tokens, lemmas, POS tags, and dependency relations. This corpus provides approximately 1,000 sentences of natural Albanian text.

**3. Run the Analysis**

**Option A: Use the Jupyter Notebook**
```bash
jupyter notebook Albanian_NLP_Analysis.ipynb
```

**Option B: Run the Python Script**
```bash
cd src
python main.py
```

The default configuration uses `data/sq_tsa-ud-test.conllu` for analysis.

## 📊 Analysis Steps for Albanian

The project follows these steps as per requirements:

### Step 1: Albanian Corpus Selection
- Selected **Albanian (sq_tsa)** from Universal Dependencies
- Understanding the Albanian .conllu file structure
- Parse Albanian corpus using the `conllu` library

### Step 2: Albanian Data Loading and Preprocessing
- Load Albanian .conllu files with `ud_loader.py`
- Extract Albanian tokens, lemmas, and PoS tags
- Structure Albanian data for analysis

### Step 3: Albanian Corpus Statistics
- Count total Albanian sentences and tokens
- Compute Albanian vocabulary size (unique lemmas)
- Calculate PoS tag frequency distribution for Albanian
- Visualize Albanian linguistic distributions using bar and pie charts

### Step 4: Albanian Text Processing Function
- Implement tokenization for Albanian text using NLTK
- Apply lemmatization and stemming to Albanian words
- Process new Albanian sentences with the `TextPreprocessor` class

### Step 5: TF-IDF Vectorization on Albanian
- Convert Albanian corpus subset to TF-IDF vectors
- Extract feature weights for Albanian terms

### Step 6: Albanian Similarity Analysis
- Compute cosine similarity matrix for Albanian sentences
- Calculate Euclidean distance matrix
- Compare Albanian sentence pairs

### Step 7: Find Similar Albanian Sentences
- Identify most similar Albanian sentence pairs
- Rank by similarity scores
- Generate comparative analysis of Albanian texts

## 📈 Output Files

The analysis generates the following files:

### Reports (CSV)
- `corpus_statistics.csv` - Overall corpus statistics
- `pos_distribution.csv` - PoS tag frequencies
- `top_frequent_words.csv` - Most frequent words
- `top_frequent_lemmas.csv` - Most frequent lemmas
- `most_similar_pairs.csv` - Top similar sentence pairs

### Visualizations (PNG)
- `pos_distribution.png` - PoS tag bar chart and pie chart
- `sentence_length_distribution.png` - Sentence length histogram
- `similarity_distribution.png` - Similarity score distributions

## 🔧 Module Documentation

### `ud_loader.py`
- `load_conllu_file(file_path)` - Load and parse .conllu files
- `extract_sentence_data(sentences)` - Extract linguistic annotations
- `get_conllu_sample(sentences, n)` - Get sample sentences

### `corpus_statistics.py`
- `compute_corpus_statistics(corpus_data)` - Calculate Albanian corpus stats
- `compute_pos_distribution(pos_tags)` - Albanian PoS frequency distribution
- `get_top_frequent_words(tokens, n)` - Most frequent Albanian words
- `create_statistics_summary(stats)` - Format Albanian statistics table

### `preprocessor.py`
- `TextPreprocessor` class - Text preprocessing utilities
  - `process_sentence(text)` - Tokenize and lemmatize
  - `get_lemmas(text)` - Extract lemmas
  - `get_stems(text)` - Extract stems

### `similarity.py`
- `SimilarityAnalyzer` class - TF-IDF and similarity analysis
  - `fit_transform(sentences)` - Create TF-IDF matrix
  - `compute_cosine_similarity()` - Calculate cosine similarity
  - `compute_euclidean_distance()` - Calculate Euclidean distance
  - `find_most_similar_pairs(n)` - Find top similar pairs

### `visualizer.py`
- `plot_pos_distribution()` - Visualize PoS tags
- `plot_similarity_distribution()` - Visualize similarity scores
- `plot_sentence_length_distribution()` - Visualize sentence lengths
- `plot_top_frequent_words()` - Visualize word frequencies

## 📝 Example Usage

```python
from src.ud_loader import load_conllu_file, extract_sentence_data
from src.corpus_statistics import compute_corpus_statistics, compute_pos_distribution
from src.preprocessor import TextPreprocessor
from src.similarity import SimilarityAnalyzer

# Load Albanian corpus
sentences = load_conllu_file('data/sq_tsa-ud-test.conllu')
corpus_data = extract_sentence_data(sentences)

# Compute Albanian statistics
stats = compute_corpus_statistics(corpus_data)
pos_df = compute_pos_distribution(corpus_data['all_pos_tags'])

# Process new Albanian text
preprocessor = TextPreprocessor()
result = preprocessor.process_sentence("Përpunimi i gjuhës natyrore është shumë interesant.")
print(result['processed'])  # Lemmatized Albanian tokens

# Similarity analysis on Albanian sentences
analyzer = SimilarityAnalyzer()
tfidf_matrix = analyzer.fit_transform(corpus_data['sentence_texts'][:1000])
cosine_sim = analyzer.compute_cosine_similarity()
similar_pairs = analyzer.find_most_similar_pairs(corpus_data['sentence_texts'][:1000], 
                                                  cosine_sim, top_n=10)
```

## 📦 Dependencies

- **conllu** (4.5.3) - Parse Universal Dependencies files
- **pandas** (2.1.4) - Data manipulation
- **numpy** (1.26.2) - Numerical computing
- **nltk** (3.8.1) - Natural language toolkit
- **scikit-learn** (1.3.2) - Machine learning and TF-IDF
- **matplotlib** (3.8.2) - Plotting
- **seaborn** (0.13.0) - Statistical visualizations
- **jupyter** (1.0.0) - Interactive notebooks

## 📸 Visualizations

Run the notebook or script to generate Albanian-specific analysis:
1. Albanian PoS distribution charts
2. Similarity heatmaps for Albanian sentence pairs
3. Albanian sentence length distributions
4. Albanian corpus statistical summaries

## 📄 Final Report Checklist

Your final report should include:

- [x] **Selected Language and Dataset**: Albanian (sq_tsa) UD treebank
- [x] **Processing Steps**: Explain each step of the Albanian analysis
- [x] **Corpus Statistics**: Present tables with Albanian sentences, tokens, vocabulary, PoS distribution
- [x] **Python Code**: Include the notebook or script with reproducible Albanian analysis code
- [x] **Screenshots**: Capture Albanian visualizations and execution results

## 🎓 Educational Notes

### Understanding Albanian .conllu Format

Each line in the Albanian .conllu file represents a token with:
- **ID**: Token index
- **FORM**: Albanian word form
- **LEMMA**: Albanian lemma/dictionary form
- **UPOS**: Universal PoS tag
- **XPOS**: Albanian-specific PoS tag
- **FEATS**: Albanian morphological features
- **HEAD**: Dependency relation head
- **DEPREL**: Dependency relation type

### Albanian PoS Tags

Common PoS tags found in Albanian corpus:
- **NOUN**: Albanian nouns (emër)
- **VERB**: Albanian verbs (folje)
- **ADJ**: Albanian adjectives (mbiemër)
- **ADV**: Albanian adverbs (ndajfolje)
- **PRON**: Albanian pronouns (përemër)
- **DET**: Albanian determiners
- **ADP**: Albanian adpositions (parafjalë)
- **CONJ**: Albanian conjunctions (lidhëz)
- **PUNCT**: Punctuation

### About Albanian Language

Albanian (shqip) is an Indo-European language with approximately 7.5 million speakers. The UD Albanian TSA treebank represents **spoken Albanian**, making it particularly valuable for studying colloquial language patterns and conversational structures.

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) measures word importance:
- **TF**: How often a word appears in a document
- **IDF**: How rare the word is across all documents
- **TF-IDF**: TF × IDF (higher for important, distinctive words)

### Similarity Metrics

- **Cosine Similarity**: Measures angle between vectors (0 to 1, higher = more similar)
- **Euclidean Distance**: Straight-line distance between vectors (lower = more similar)

## 🤝 Contributing

Feel free to extend this Albanian NLP project by:
- Adding more similarity metrics (Jaccard, Dice coefficient)
- Implementing Albanian dependency parsing visualization
- Comparing Albanian with other Balkan languages
- Creating interactive dashboards for Albanian linguistic features
- Adding Albanian-specific NLP tools and resources

## 📚 References

- [Universal Dependencies](https://universaldependencies.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

## 📧 Contact

For questions or issues, please refer to the course materials or contact your instructor.

---

**Happy Analyzing! 🎉**
