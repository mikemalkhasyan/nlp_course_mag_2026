"""
LDA Topic Modeling - Part 2: Topic Labeling Script
===================================================
This script allows you to assign meaningful names to discovered topics.

Steps:
1. Load the trained model and dictionary
2. Display topics with their top words
3. Prompt user to name each topic
4. Save topic labels to JSON file
"""

import os
import json
from gensim.models import LdaModel
from gensim import corpora


# STEP 1: LOAD MODEL AND DICTIONARY

def load_model_and_dictionary():
    """
    Load the trained LDA model and dictionary from the models directory.
    
    Returns:
        tuple: (lda_model, dictionary)
    """
    models_dir = "models"
    
    print("Loading trained model and dictionary...")
    
    # Check if files exist
    model_path = os.path.join(models_dir, "lda_model")
    dict_path = os.path.join(models_dir, "dictionary.dict")
    
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'")
        print("Please run '1_training.py' first to train the model.")
        return None, None
    
    if not os.path.exists(dict_path):
        print(f"Dictionary file not found at '{dict_path}'")
        print("Please run '1_training.py' first to train the model.")
        return None, None
    
    # Load model and dictionary
    lda_model = LdaModel.load(model_path)
    dictionary = corpora.Dictionary.load(dict_path)
    
    print(f"✓ Loaded model with {lda_model.num_topics} topics")
    print(f"✓ Loaded dictionary with {len(dictionary)} words\n")
    
    return lda_model, dictionary


# STEP 2: DISPLAY TOPIC DETAILS

def display_topic_details(lda_model, topic_id, num_words=20):
    """
    Display detailed information about a specific topic.
    
    Args:
        lda_model: Trained LDA model
        topic_id: ID of the topic to display
        num_words: Number of top words to show (default: 20)
    """
    print(f"\n{'='*80}")
    print(f"TOPIC {topic_id}")
    print('='*80)
    
    # Get top words with probabilities
    top_words = lda_model.show_topic(topic_id, topn=num_words)
    
    # Display in a formatted table
    print(f"{'Rank':<6} {'Word':<20} {'Probability':<12}")
    print('-'*80)
    
    for rank, (word, prob) in enumerate(top_words, 1):
        print(f"{rank:<6} {word:<20} {prob:.6f}")
    
    print('='*80)


# STEP 3: COLLECT TOPIC LABELS

def collect_topic_labels(lda_model):
    """
    Interactively collect labels for each topic from the user.
    
    Args:
        lda_model: Trained LDA model
    
    Returns:
        dict: Dictionary mapping topic_id to topic_name
    """
    print("\n" + "="*80)
    print("TOPIC LABELING")
    print("="*80)
    print("\nInstructions:")
    print("- Review each topic's top words carefully")
    print("- Enter a meaningful name (e.g., 'Technology', 'Sports', 'Politics')")
    print("- Press Enter to skip naming a topic (keeps default name 'Topic N')")
    
    input("Press Enter to start labeling topics...")
    
    topic_labels = {}
    
    # Iterate through each topic
    for topic_id in range(lda_model.num_topics):
        # Display topic details
        display_topic_details(lda_model, topic_id, num_words=20)
        
        # Prompt for label
        while True:
            label = input(f"\nEnter a name for Topic {topic_id} (or press Enter to skip): ").strip()
            
            # If user pressed Enter, use default name
            if not label:
                label = f"Topic {topic_id}"
                print(f"Using default name: '{label}'")
                break
            
            # If user entered a name, use it
            if label:
                print(f"✓ Topic {topic_id} labeled as: '{label}'")
                break
        
        topic_labels[str(topic_id)] = label  # Use string key for JSON compatibility
    
    return topic_labels


# STEP 4: SAVE TOPIC LABELS

def save_topic_labels(topic_labels):
    """
    Save topic labels to a JSON file.
    
    Args:
        topic_labels: Dictionary mapping topic_id to topic_name
    """
    models_dir = "models"
    labels_path = os.path.join(models_dir, "topic_labels.json")
    
    with open(labels_path, 'w') as f:
        json.dump(topic_labels, f, indent=4)
    
    print(f"\n✓ Topic labels saved to '{labels_path}'")


# STEP 5: DISPLAY SUMMARY

def display_summary(topic_labels):
    """
    Display a summary of all topics with their assigned names.
    
    Args:
        topic_labels: Dictionary mapping topic_id to topic_name
    """
    print("\n" + "="*80)
    print("TOPIC LABELING SUMMARY")
    print("="*80)
    
    for topic_id, topic_name in sorted(topic_labels.items(), key=lambda x: int(x[0])):
        print(f"Topic {topic_id}: {topic_name}")
    
    print("="*80)


# MAIN EXECUTION

def main():
    """
    Main function: orchestrates the topic labeling process.
    """
    print("\n" + "="*80)
    print("LDA TOPIC MODELING - TOPIC LABELING")
    print("="*80 + "\n")
    
    # Step 1: Load model and dictionary
    lda_model, dictionary = load_model_and_dictionary()
    
    if lda_model is None or dictionary is None:
        return
    
    # Step 2 & 3: Display topics and collect labels
    topic_labels = collect_topic_labels(lda_model)
    
    # Step 4: Save labels
    save_topic_labels(topic_labels)
    
    # Step 5: Display summary
    display_summary(topic_labels)
    
    print("\n✓ Labeling complete!")


if __name__ == "__main__":
    main()