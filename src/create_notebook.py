"""
Script to create the main analysis Jupyter notebook
"""

import json

# Create the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells
cells = [
    # Title cell
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NLP Processing and Analysis with Universal Dependencies (UD)\\n",
            "\\n",
            "**Project Overview:**\\n",
            "This notebook demonstrates comprehensive NLP analysis using Universal Dependencies (UD) treebanks.\\n",
            "\\n",
            "**Steps covered:**\\n",
            "1. Corpus Selection and Data Loading\\n",
            "2. Data Preprocessing and Extraction\\n",
            "3. Corpus Statistics\\n",
            "4. PoS Tag Distribution and Visualization\\n",
            "5. Custom Sentence Processing\\n",
            "6. TF-IDF Vectorization\\n",
            "7. Similarity Analysis\\n",
            "8. Most Similar Sentence Pairs"
        ]
    },
    # Imports
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Import Required Libraries"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# System\\n",
            "import sys\\n",
            "from pathlib import Path\\n",
            "import warnings\\n",
            "warnings.filterwarnings('ignore')\\n",
            "\\n",
            "# Add src to path\\n",
            "sys.path.append(str(Path.cwd().parent / 'src'))\\n",
            "\\n",
            "# Core libraries\\n",
            "import pandas as pd\\n",
            "import numpy as np\\n",
            "from tqdm import tqdm\\n",
            "\\n",
            "# Custom modules\\n",
            "from ud_loader import load_conllu_file, extract_sentence_data, get_conllu_sample\\n",
            "from statistics import (compute_corpus_statistics, compute_pos_distribution,\\n",
            "                       create_statistics_summary, get_top_frequent_words,\\n",
            "                       get_top_frequent_lemmas)\\n",
            "from preprocessor import TextPreprocessor\\n",
            "from similarity import SimilarityAnalyzer\\n",
            "from visualizer import (plot_pos_distribution, plot_similarity_distribution,\\n",
            "                       plot_sentence_length_distribution, plot_top_frequent_words)\\n",
            "\\n",
            "print('✓ All libraries imported successfully!')"
        ]
    }
]

# Write the notebook
output_path = "../notebooks/ud_analysis.ipynb"
notebook["cells"] = cells

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Notebook created: {output_path}")
print("Note: This is a starter. Add more cells manually or run the full main.py script.")
