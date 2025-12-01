# Quick Start Guide

## Setup Instructions

### 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- conllu (for parsing UD files)
- pandas, numpy (data manipulation)
- nltk (NLP tools)
- scikit-learn (TF-IDF and similarity)
- matplotlib, seaborn (visualizations)
- jupyter (notebooks)

### 2. Download NLTK Data

After installing, download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

Or run this command:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

### 3. Download Universal Dependencies Data

1. Visit: https://universaldependencies.org/#download
2. Choose a language (examples below)
3. Download the treebank and extract
4. Copy the `.conllu` file to the `data/` folder

**Popular choices:**

| Language | Treebank | File Name |
|----------|----------|-----------|
| English | UD_English-EWT | `en_ewt-ud-train.conllu` |
| Spanish | UD_Spanish-GSD | `es_gsd-ud-train.conllu` |
| German | UD_German-GSD | `de_gsd-ud-train.conllu` |
| French | UD_French-GSD | `fr_gsd-ud-train.conllu` |
| Italian | UD_Italian-ISDT | `it_isdt-ud-train.conllu` |
| Portuguese | UD_Portuguese-Bosque | `pt_bosque-ud-train.conllu` |

### 4. Run the Analysis

**Option A: Python Script**

1. Navigate to the src folder:
   ```bash
   cd src
   ```

2. Edit `main.py` and update line 20:
   ```python
   CONLLU_FILE = "../data/YOUR_FILE_HERE.conllu"  # Change this
   LANGUAGE = "Your Language"  # Change this
   ```

3. Run the script:
   ```bash
   python main.py
   ```

**Option B: Jupyter Notebook**

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/ud_analysis.ipynb`

3. Update the configuration cell with your file path

4. Run all cells (Cell → Run All)

## Troubleshooting

### Module Not Found

If you get `ModuleNotFoundError`:
```bash
pip install -r requirements.txt
```

### NLTK Data Not Found

If you get NLTK data errors:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### .conllu File Not Found

Make sure:
1. You downloaded a .conllu file
2. It's placed in the `data/` folder
3. The path in the code is correct (use forward slashes `/` or double backslashes `\\\\`)

### Import Errors in Notebook

Add this to the first cell:
```python
import sys
sys.path.append('../src')
```

## Expected Output

After running successfully, you'll see:

### Console Output:
- Corpus loading status
- Statistics summary table
- PoS distribution
- Top frequent words/lemmas
- Similarity analysis results
- File generation confirmations

### Generated Files:

**Reports (CSV):**
- `reports/corpus_statistics.csv`
- `reports/pos_distribution.csv`
- `reports/top_frequent_words.csv`
- `reports/top_frequent_lemmas.csv`
- `reports/most_similar_pairs.csv`

**Visualizations (PNG):**
- `outputs/pos_distribution.png`
- `outputs/sentence_length_distribution.png`
- `outputs/similarity_distribution.png`

## Example Run Time

- Small corpus (< 5,000 sentences): ~30 seconds
- Medium corpus (5,000-15,000 sentences): 1-3 minutes
- Large corpus (> 15,000 sentences): 3-10 minutes

## Testing Installation

Run the test script to verify everything is installed:

```bash
cd src
python test_modules.py
```

This will test all modules without requiring a .conllu file.

## Next Steps

1. ✅ Install dependencies
2. ✅ Download NLTK data
3. ✅ Download .conllu file
4. ✅ Run analysis
5. ✅ Check generated reports and visualizations
6. ✅ Include results in your project report

## For Your Report

Include:
1. Screenshots of visualizations
2. Tables from the CSV reports
3. Sample code snippets
4. Explanation of each analysis step
5. Interpretation of results

Good luck with your project! 🎓
