"""
LDA Topic Modeling - Part 4: Visualizations
============================================
This script automatically creates all visualizations for LDA model.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import LdaModel
from gensim import corpora
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from gensim.parsing.preprocessing import STOPWORDS

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_resources():
    """Load model, dictionary, and labels."""
    models_dir = "models"
    
    lda_model = LdaModel.load(os.path.join(models_dir, "lda_model"))
    dictionary = corpora.Dictionary.load(os.path.join(models_dir, "dictionary.dict"))
    
    labels_path = os.path.join(models_dir, "topic_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            topic_labels = json.load(f)
    else:
        topic_labels = {str(i): f"Topic {i}" for i in range(lda_model.num_topics)}
    
    return lda_model, dictionary, topic_labels


def viz_topic_word_heatmap(lda_model, topic_labels, top_n=15):
    """Create topic-word distribution heatmap."""
    all_words = set()
    topic_word_probs = {}
    
    for topic_id in range(lda_model.num_topics):
        top_words = lda_model.show_topic(topic_id, topn=top_n)
        topic_word_probs[topic_id] = dict(top_words)
        all_words.update([word for word, _ in top_words])
    
    words_list = sorted(all_words)
    matrix = np.zeros((lda_model.num_topics, len(words_list)))
    
    for topic_id in range(lda_model.num_topics):
        for word_idx, word in enumerate(words_list):
            matrix[topic_id, word_idx] = topic_word_probs[topic_id].get(word, 0)
    
    plt.figure(figsize=(16, 10))
    topic_names = [topic_labels.get(str(i), f"Topic {i}") for i in range(lda_model.num_topics)]
    
    sns.heatmap(matrix, xticklabels=words_list, yticklabels=topic_names,
                cmap='YlOrRd', cbar_kws={'label': 'Probability'}, linewidths=0.5)
    
    plt.title('Topic-Word Distribution Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Words', fontsize=12, fontweight='bold')
    plt.ylabel('Topics', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = 'visualizations/topic_word_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path}")


def viz_top_words_bars(lda_model, topic_labels, top_n=10):
    """Create bar charts for top words per topic."""
    num_topics = lda_model.num_topics
    fig, axes = plt.subplots(nrows=(num_topics + 1) // 2, ncols=2, 
                            figsize=(16, 4 * ((num_topics + 1) // 2)))
    axes = axes.flatten()
    
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=top_n)
        words = [word for word, _ in top_words]
        probs = [prob for _, prob in top_words]
        
        ax = axes[topic_id]
        ax.barh(words, probs, color=plt.cm.viridis(topic_id / num_topics))
        
        topic_name = topic_labels.get(str(topic_id), f"Topic {topic_id}")
        ax.set_title(f'{topic_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Probability', fontsize=10)
        ax.invert_yaxis()
    
    for idx in range(num_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top Words per Topic', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'visualizations/top_words_per_topic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path}")

def viz_document_topics(lda_model, dictionary, topic_labels, sample_size=100):
    """Visualize -topic distribution."""
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:sample_size]
    
    topic_counts = Counter()
    doc_topics = []
    
    for doc in documents:
        tokens = doc.lower().split()
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) >= 3 and word.isalpha()]
        bow = dictionary.doc2bow(tokens)
        topic_dist = lda_model.get_document_topics(bow)
        
        if topic_dist:
            dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
            topic_counts[dominant_topic] += 1
            doc_topics.append(topic_dist)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    topics = sorted(topic_counts.keys())
    counts = [topic_counts[t] for t in topics]
    topic_names = [topic_labels.get(str(t), f"Topic {t}") for t in topics]
    colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))
    
    ax1.bar(range(len(topics)), counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(topics)))
    ax1.set_xticklabels(topic_names, rotation=45, ha='right')
    ax1.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
    ax1.set_title(f'Dominant Topic Distribution', fontsize=14, fontweight='bold')
    ax1.grid(False)
    
    if doc_topics:
        num_docs = min(50, len(doc_topics))
        matrix = np.zeros((num_docs, lda_model.num_topics))
        
        for doc_idx, topic_dist in enumerate(doc_topics[:num_docs]):
            for topic_id, prob in topic_dist:
                matrix[doc_idx, topic_id] = prob
        
        im = ax2.imshow(matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax2.set_yticks(range(lda_model.num_topics))
        ax2.set_yticklabels([topic_labels.get(str(i), f"T{i}") for i in range(lda_model.num_topics)])
        ax2.set_xlabel('Document Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Topics', fontsize=12, fontweight='bold')
        ax2.set_title('Document-Topic Matrix', fontsize=14, fontweight='bold')
        ax2.grid(False)
        plt.colorbar(im, ax=ax2, label='Probability')
    
    plt.tight_layout()
    
    output_path = 'visualizations/document_topic_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path}")


def viz_wordclouds(lda_model, topic_labels):
    """Create word clouds for topics."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠ Skipping word clouds (install: pip install wordcloud)")
        return
    
    num_topics = lda_model.num_topics
    fig, axes = plt.subplots(nrows=(num_topics + 1) // 2, ncols=2,
                            figsize=(16, 6 * ((num_topics + 1) // 2)))
    axes = axes.flatten()
    
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=50)
        word_freq = {word: prob for word, prob in top_words}
        
        wc = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', relative_scaling=0.5).generate_from_frequencies(word_freq)
        
        ax = axes[topic_id]
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        topic_name = topic_labels.get(str(topic_id), f"Topic {topic_id}")
        ax.set_title(f'{topic_name}', fontsize=14, fontweight='bold', pad=10)
    
    for idx in range(num_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Topic Word Clouds', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'visualizations/topic_wordclouds.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path}")

def main():
    print("\n" + "="*80)
    print("GENERATING LDA VISUALIZATIONS")
    print("="*80 + "\n")
    
    lda_model, dictionary, topic_labels = load_resources()
    os.makedirs('visualizations', exist_ok=True)
    
    viz_topic_word_heatmap(lda_model, topic_labels)
    viz_top_words_bars(lda_model, topic_labels)
    viz_document_topics(lda_model, dictionary, topic_labels)
    viz_wordclouds(lda_model, topic_labels)
    
    print("\n" + "="*80)
    print("✓ All visualizations saved to: visualizations/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()