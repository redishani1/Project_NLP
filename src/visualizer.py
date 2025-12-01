"""
Module for creating visualizations of corpus statistics and similarity analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict


# Set default style
plt.style.use('ggplot')
sns.set_palette("husl")


def plot_pos_distribution(pos_df: pd.DataFrame, language: str = "Corpus", 
                         save_path: str = None):
    """
    Create bar chart and pie chart for PoS tag distribution.
    
    Args:
        pos_df: DataFrame with PoS tag frequencies
        language: Name of the language/corpus
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    ax1 = axes[0]
    colors = sns.color_palette('husl', len(pos_df))
    bars = ax1.bar(pos_df['PoS Tag'], pos_df['Frequency'], color=colors)
    ax1.set_xlabel('Part-of-Speech Tags', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'PoS Tag Distribution - {language}', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)
    
    # Pie chart (top 10)
    ax2 = axes[1]
    top_n = 10
    top_pos = pos_df.head(top_n)
    other_sum = pos_df.iloc[top_n:]['Frequency'].sum()
    
    if other_sum > 0:
        plot_data = pd.concat([top_pos, pd.DataFrame([{'PoS Tag': 'Others', 'Frequency': other_sum}])])
    else:
        plot_data = top_pos
    
    wedges, texts, autotexts = ax2.pie(plot_data['Frequency'], 
                                        labels=plot_data['PoS Tag'],
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=sns.color_palette('Set3', len(plot_data)))
    ax2.set_title(f'Top {top_n} PoS Tags - {language}', fontsize=14, fontweight='bold')
    
    # Make text more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_similarity_distribution(similarity_values: np.ndarray, save_path: str = None):
    """
    Create histogram and box plot for similarity distribution.
    
    Args:
        similarity_values: Array of similarity values
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(similarity_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Pairwise Cosine Similarities', fontsize=14, fontweight='bold')
    ax1.axvline(similarity_values.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {similarity_values.mean():.3f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    box_parts = ax2.boxplot([similarity_values], vert=True, patch_artist=True,
                            labels=['Cosine Similarity'],
                            boxprops=dict(facecolor='lightgreen', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5))
    ax2.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
    ax2.set_title('Cosine Similarity Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"Min: {similarity_values.min():.3f}\n"
    stats_text += f"Q1: {np.percentile(similarity_values, 25):.3f}\n"
    stats_text += f"Median: {np.median(similarity_values):.3f}\n"
    stats_text += f"Q3: {np.percentile(similarity_values, 75):.3f}\n"
    stats_text += f"Max: {similarity_values.max():.3f}"
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sentence_length_distribution(sent_lengths: List[int], 
                                      language: str = "Corpus",
                                      save_path: str = None):
    """
    Plot sentence length distribution.
    
    Args:
        sent_lengths: List of sentence lengths
        language: Name of the language/corpus
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(sent_lengths, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Sentence Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Sentence Length Distribution - {language}', fontsize=14, fontweight='bold')
    
    # Add mean and median lines
    mean_len = np.mean(sent_lengths)
    median_len = np.median(sent_lengths)
    ax.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
    ax.axvline(median_len, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_top_frequent_words(freq_df: pd.DataFrame, title: str = "Top Frequent Words",
                           save_path: str = None):
    """
    Plot horizontal bar chart of most frequent words.
    
    Args:
        freq_df: DataFrame with word frequencies
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by frequency and plot
    data = freq_df.sort_values('Frequency', ascending=True)
    colors = sns.color_palette('viridis', len(data))
    
    bars = ax.barh(data.iloc[:, 1], data['Frequency'], color=colors)
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Word/Lemma', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{int(width):,}',
               ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
