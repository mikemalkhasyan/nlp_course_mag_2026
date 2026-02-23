"""
LDA Topic Modeling - Part 1: Training Script
=============================================
This script trains an LDA model to discover hidden topics in documents.

Steps:
1. Load the 20 Newsgroups dataset
2. Preprocess the text (clean, tokenize, remove stopwords)
3. Create dictionary and corpus
4. Train LDA model
5. Save model and dictionary
6. Display discovered topics
"""

import os
from sklearn.datasets import fetch_20newsgroups
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

# STEP 1: LOAD DATASET

def load_dataset():
    """
    Load the 20 Newsgroups dataset.
    
    Args:
        num_docs: Number of documents to use (default: 1000)
    
    Returns:
        List of document texts
    """
    import os
    from pathlib import Path
    
    num_docs = None
    if num_docs:
        print(f"Loading {num_docs} documents from 20 Newsgroups dataset...")
    else:
        print("Loading all documents from 20 Newsgroups dataset...")

    # Check if dataset is already cached
    # sklearn caches data in ~/scikit_learn_data/ by default
    cache_dir = Path.home() / 'scikit_learn_data' / '20news_home'
    
    if cache_dir.exists():
        print("✓ Found cached dataset (no download needed)")
    else:
        print("Dataset not found in cache. Downloading...")
    
    # Load dataset with headers, footers, and quotes removed
    # fetch_20newsgroups handles caching automatically
    newsgroups = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        download_if_missing=True  # Download only if not cached
    )
    
    # Take only the first 'num_docs' documents
    if num_docs:
        documents = newsgroups.data[:num_docs]
    else:
        documents = newsgroups.data
    
    print(f"✓ Loaded {len(documents)} documents")
    return documents


# STEP 2: PREPROCESS TEXT

def preprocess_text(documents):
    """
    Preprocess documents: tokenize, lowercase, remove stopwords, filter words.
    
    Preprocessing steps:
    - Tokenize into words
    - Convert to lowercase
    - Remove stopwords
    - Keep only words with 3+ characters
    
    Args:
        documents: List of document texts
    
    Returns:
        List of tokenized documents (list of lists of words)
    """
    print("\nPreprocessing documents...")
    
    processed_docs = []
    
    for doc in documents:
        # Tokenize: split into words
        tokens = doc.lower().split()
        
        # Remove stopwords and short words
        tokens = [
            word for word in tokens
            if word not in STOPWORDS  # Remove common words (the, is, at, etc.)
            and len(word) >= 3        # Keep only words with 3+ characters
            and word.isalpha()        # Keep only alphabetic words (no numbers)
        ]
        
        processed_docs.append(tokens)
    
    print(f"✓ Preprocessed {len(processed_docs)} documents")
    
    # Show example
    if processed_docs:
        print(f"\nExample preprocessed document (first 10 words):")
        print(processed_docs[0][:10])
    
    return processed_docs


# STEP 3: CREATE DICTIONARY AND CORPUS

def create_dictionary_and_corpus(processed_docs):
    """
    Create dictionary (vocabulary) and corpus (bag-of-words representation).
    
    Filtering:
    - Remove words appearing in < 5 documents (too rare)
    - Remove words appearing in > 50% of documents (too common)
    
    Args:
        processed_docs: List of tokenized documents
    
    Returns:
        tuple: (dictionary, corpus)
    """
    print("\nCreating dictionary and corpus...")
    
    # Create dictionary: assigns unique ID to each word
    dictionary = corpora.Dictionary(processed_docs)
    
    print(f"Dictionary created with {len(dictionary)} unique words")
    
    # Filter extremes
    dictionary.filter_extremes(
        no_below=5,        # Remove words appearing in < 5 documents
        no_above=0.5       # Remove words appearing in > 50% of documents
    )
    
    print(f"After filtering: {len(dictionary)} words remaining")
    
    # Create corpus: bag-of-words representation
    # Each document is converted to list of (word_id, word_count) tuples
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    print(f"✓ Created corpus with {len(corpus)} documents")
    
    return dictionary, corpus


# STEP 4: TRAIN LDA MODEL

def train_lda_model(corpus, dictionary, num_topics=10, passes=15):
    """
    Train the LDA model.
    
    Args:
        corpus: Bag-of-words corpus
        dictionary: Gensim dictionary
        num_topics: Number of topics to discover (default: 10)
        passes: Number of training iterations (default: 15)
    
    Returns:
        Trained LDA model
    """
    print(f"\nTraining LDA model with {num_topics} topics...")
    print(f"Using {passes} passes through the corpus")
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        # iterations=100000,
        passes=passes,
        alpha='auto',      # Learn document-topic distribution automatically
        eta='auto',        # Learn topic-word distribution automatically
        random_state=42    # For reproducibility
    )
    
    print("✓ Model training complete!")
    
    return lda_model


# STEP 5: SAVE MODEL AND DICTIONARY

def save_models(lda_model, dictionary):
    """
    Save the trained model and dictionary to the 'models' directory.
    
    Args:
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
    """
    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"\n✓ Created '{models_dir}' directory")
    
    # Save model and dictionary
    lda_model.save(os.path.join(models_dir, "lda_model"))
    dictionary.save(os.path.join(models_dir, "dictionary.dict"))
    
    print(f"✓ Model saved to '{models_dir}/lda_model'")
    print(f"✓ Dictionary saved to '{models_dir}/dictionary.dict'")


# STEP 6: DISPLAY TOPICS

def display_topics(lda_model, num_words=15):
    """
    Display all discovered topics with their top words.
    
    Args:
        lda_model: Trained LDA model
        num_words: Number of top words to show per topic (default: 15)
    """
    print("\n" + "="*80)
    print(f"DISCOVERED TOPICS (Top {num_words} words per topic)")
    print("="*80)
    
    for topic_id in range(lda_model.num_topics):
        # Get top words for this topic
        top_words = lda_model.show_topic(topic_id, topn=num_words)
        
        # Format words and probabilities
        words = [f"{word} ({prob:.3f})" for word, prob in top_words]
        
        print(f"\nTopic {topic_id}:")
        print(", ".join(words))
    
    print("\n" + "="*80)


# MAIN EXECUTION

def main():
    """
    Main function: orchestrates the entire training pipeline.
    """
    print("\n" + "="*80)
    print("LDA TOPIC MODELING - TRAINING")
    print("="*80)
    
    # Step 1: Load data
    documents = load_dataset()
    
    # Step 2: Preprocess
    processed_docs = preprocess_text(documents)
    
    # Step 3: Create dictionary and corpus
    dictionary, corpus = create_dictionary_and_corpus(processed_docs)
    
    # Step 4: Train model
    lda_model = train_lda_model(corpus, dictionary, num_topics=7, passes=30)
    
    # Step 5: Save model
    save_models(lda_model, dictionary)
    
    # Step 6: Display topics
    display_topics(lda_model, num_words=15)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()