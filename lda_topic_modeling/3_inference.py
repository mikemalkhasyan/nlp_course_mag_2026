"""
LDA Topic Modeling - Part 3: Inference Script
==============================================
This script classifies new documents using the trained model.

Steps:
1. Load model, dictionary, and topic labels
2. Display topic summary
3. Classify sample documents
4. Display results with topic names and probabilities
"""

import os
import json
from gensim.models import LdaModel
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS


# SAMPLE DOCUMENTS FOR TESTING

SAMPLE_DOCUMENTS = {
    "Sample 1 - Gaming": """
        The new graphics card delivers amazing performance for gaming. The GPU 
        can handle 4K resolution easily with ray tracing enabled. Gamers will 
        love the improved frame rates.
    """,
    
    "Sample 2 - Science": """
        Scientists discovered a new exoplanet orbiting a distant star in the 
        habitable zone. The research team published their findings in Nature 
        journal. This discovery could provide insights into planetary formation.
    """,
    
    "Sample 3 - Sports": """
        The basketball team won the championship after an incredible final game. 
        The players celebrated with fans in the stadium. It was the team's first 
        title in twenty years.
    """,
    
    "Sample 4 - Politics": """
        Congress passed a new bill regarding healthcare reform. The president 
        is expected to sign the legislation next week. The policy will affect 
        millions of citizens across the country.
    """,
    
    "Sample 5 - Food": """
        I love cooking Italian food at home. Pasta carbonara and margherita 
        pizza are my favorite dishes to make. Fresh ingredients make all the 
        difference in authentic recipes.
    """
}


# STEP 1: LOAD MODEL, DICTIONARY, AND LABELS

def load_resources():
    """
    Load the trained model, dictionary, and topic labels.
    
    Returns:
        tuple: (lda_model, dictionary, topic_labels)
    """
    models_dir = "models"
    
    print("Loading resources...")
    
    # Check if files exist
    model_path = os.path.join(models_dir, "lda_model")
    dict_path = os.path.join(models_dir, "dictionary.dict")
    labels_path = os.path.join(models_dir, "topic_labels.json")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found. Please run '1_training.py' first.")
        return None, None, None
    
    if not os.path.exists(dict_path):
        print(f"❌ Error: Dictionary not found. Please run '1_training.py' first.")
        return None, None, None
    
    # Load model and dictionary
    lda_model = LdaModel.load(model_path)
    dictionary = corpora.Dictionary.load(dict_path)
    
    # Load topic labels (if they exist)
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            topic_labels = json.load(f)
        print("✓ Loaded topic labels")
    else:
        # If labels don't exist, create default labels
        topic_labels = {str(i): f"Topic {i}" for i in range(lda_model.num_topics)}
        print("⚠ No topic labels found. Using default names.")
        print("  Run '2_labeling.py' to assign meaningful names to topics.")
    
    print(f"✓ Loaded model with {lda_model.num_topics} topics")
    print(f"✓ Loaded dictionary with {len(dictionary)} words\n")
    
    return lda_model, dictionary, topic_labels


# STEP 2: DISPLAY TOPIC SUMMARY

def display_topic_summary(lda_model, topic_labels):
    """
    Display a summary of all available topics.
    
    Args:
        lda_model: Trained LDA model
        topic_labels: Dictionary of topic labels
    """
    print("="*80)
    print("AVAILABLE TOPICS")
    print("="*80)
    
    for topic_id in range(lda_model.num_topics):
        topic_name = topic_labels.get(str(topic_id), f"Topic {topic_id}")
        top_words = lda_model.show_topic(topic_id, topn=5)
        words = [word for word, _ in top_words]
        
        print(f"\nTopic {topic_id}: {topic_name}")
        print(f"  Key words: {', '.join(words)}")
    
    print("\n" + "="*80 + "\n")


# STEP 3: PREPROCESS DOCUMENT

def preprocess_document(text):
    """
    Preprocess a single document (same way as training).
    
    Args:
        text: Input document text
    
    Returns:
        List of tokens (words)
    """
    # Tokenize and clean
    tokens = text.lower().split()
    
    # Remove stopwords and short words (same as training)
    tokens = [
        word for word in tokens
        if word not in STOPWORDS
        and len(word) >= 3
        and word.isalpha()
    ]
    
    return tokens


# STEP 4: CLASSIFY DOCUMENT

def classify_document(text, lda_model, dictionary, topic_labels, top_n=3):
    """
    Classify a document and return its top topics.
    
    Args:
        text: Input document text
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
        topic_labels: Dictionary of topic labels
        top_n: Number of top topics to return (default: 3)
    
    Returns:
        List of tuples: [(topic_id, topic_name, probability, top_words), ...]
    """
    # Preprocess document
    tokens = preprocess_document(text)
    
    # Convert to bag-of-words
    bow = dictionary.doc2bow(tokens)
    
    # Get topic distribution
    topic_distribution = lda_model.get_document_topics(bow)
    
    # Sort by probability (descending)
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
    # Get top N topics
    results = []
    for topic_id, prob in sorted_topics[:top_n]:
        topic_name = topic_labels.get(str(topic_id), f"Topic {topic_id}")
        
        # Get top 5 words for this topic
        top_words = lda_model.show_topic(topic_id, topn=5)
        words = [word for word, _ in top_words]
        
        results.append((topic_id, topic_name, prob, words))
    
    return results


# STEP 5: DISPLAY CLASSIFICATION RESULTS

def display_classification(doc_name, text, classification_results):
    """
    Display the classification results in a formatted way.
    
    Args:
        doc_name: Name/identifier of the document
        text: Original document text
        classification_results: Results from classify_document()
    """
    print("="*80)
    print(f"CLASSIFICATION RESULTS: {doc_name}")
    print("="*80)
    
    # Show document preview (first 150 characters)
    preview = text.strip()[:150]
    if len(text.strip()) > 150:
        preview += "..."
    print(f"\nDocument Preview:")
    print(f"  {preview}")
    
    print(f"\nTop Topics:")
    print("-"*80)
    
    for rank, (topic_id, topic_name, prob, top_words) in enumerate(classification_results, 1):
        print(f"\n{rank}. {topic_name} (Topic {topic_id})")
        print(f"   Probability: {prob:.4f} ({prob*100:.2f}%)")
        print(f"   Key words: {', '.join(top_words)}")
    
    print("\n" + "="*80 + "\n")


# STEP 6: CLASSIFY ALL SAMPLE DOCUMENTS

def classify_sample_documents(lda_model, dictionary, topic_labels):
    """
    Classify all pre-defined sample documents.
    
    Args:
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
        topic_labels: Dictionary of topic labels
    """
    print("\n" + "="*80)
    print("CLASSIFYING SAMPLE DOCUMENTS")
    print("="*80 + "\n")
    
    for doc_name, doc_text in SAMPLE_DOCUMENTS.items():
        # Classify document
        results = classify_document(doc_text, lda_model, dictionary, topic_labels, top_n=3)
        
        # Display results
        display_classification(doc_name, doc_text, results)
        
        # Pause between documents for readability
        input("Press Enter to continue to next document...")


# STEP 7: INTERACTIVE CLASSIFICATION

def interactive_classification(lda_model, dictionary, topic_labels):
    """
    Allow user to classify their own documents interactively.
    
    Args:
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
        topic_labels: Dictionary of topic labels
    """
    print("\n" + "="*80)
    print("INTERACTIVE CLASSIFICATION MODE")
    print("="*80)
    print("\nYou can now enter your own text to classify.")
    print("Type or paste your text, then press Enter twice (empty line) to submit.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("-"*80)
        print("Enter your document (press Enter twice when done):")
        
        lines = []
        while True:
            line = input()
            if line.lower() == 'quit':
                print("\nExiting interactive mode.")
                return
            if line == '' and lines:  # Empty line after some input
                break
            if line:  # Non-empty line
                lines.append(line)
        
        if not lines:
            continue
        
        doc_text = ' '.join(lines)
        
        # Classify document
        results = classify_document(doc_text, lda_model, dictionary, topic_labels, top_n=3)
        
        # Display results
        display_classification("Your Document", doc_text, results)


# MAIN EXECUTION

def main():
    """
    Main function: orchestrates the classification process.
    """
    print("\n" + "="*80)
    print("LDA TOPIC MODELING - DOCUMENT CLASSIFICATION")
    print("="*80 + "\n")
    
    # Step 1: Load resources
    lda_model, dictionary, topic_labels = load_resources()
    
    if lda_model is None:
        return
    
    # Step 2: Display topic summary
    display_topic_summary(lda_model, topic_labels)
    
    # Menu
    while True:
        print("What would you like to do?")
        print("1. Classify sample documents (5 pre-written examples)")
        print("2. Classify your own text (interactive mode)")
        print("3. Show topic summary again")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            classify_sample_documents(lda_model, dictionary, topic_labels)
        
        elif choice == '2':
            interactive_classification(lda_model, dictionary, topic_labels)
        
        elif choice == '3':
            display_topic_summary(lda_model, topic_labels)
        
        elif choice == '4':
            print("\nThank you for using the LDA topic classifier!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()