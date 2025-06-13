#!/usr/bin/env python3
"""
Example script showing how to use the Hacker News score prediction model.
"""

from model import load_predictor
import sys
import os


def main():
    # Check if the checkpoint file exists
    checkpoint_path = "title_regression_model/cbow_final_with_vocab.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        print("Make sure the CBOW checkpoint file is in the current directory.")
        return
    
    # Load the model
    print("Loading model...")
    try:
        predictor = load_predictor(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Interactive mode if no arguments provided
    if len(sys.argv) == 1:
        print("\nInteractive mode - Enter titles to predict scores (or 'quit' to exit)")
        print("=" * 70)
        
        while True:
            try:
                title = input("\nEnter title: ").strip()
                if title.lower() in ['quit', 'exit', 'q']:
                    break
                if not title:
                    continue
                
                result = predictor.predict_single(title)
                print(f"\nResults for: '{result['title']}'")
                print(f"  Predicted Score: {result['predicted_score']:.1f}")
                print(f"  Log Score: {result['predicted_log_score']:.3f}")
                print(f"  Words in vocabulary: {result['words_in_vocab']}/{result['total_words']}")
                print(f"  Coverage: {result['coverage']:.1%}")
                
                # Show similar words for some terms
                words = title.lower().split()
                vocab_words = [w for w in words if w in predictor.word2idx]
                if vocab_words:
                    sample_word = vocab_words[0]
                    similar = predictor.get_similar_words(sample_word, top_k=5)
                    if similar:
                        print(f"  Words similar to '{sample_word}': {', '.join([w for w, _ in similar[:3]])}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Batch mode if titles provided as arguments
    else:
        titles = sys.argv[1:]
        print(f"Predicting scores for {len(titles)} title(s)...")
        print("=" * 70)
        
        results = predictor.predict_batch(titles)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result['title']}")
            print(f"   Predicted Score: {result['predicted_score']:.1f}")
            print(f"   Coverage: {result['coverage']:.1%} ({result['words_in_vocab']}/{result['total_words']} words)")
    
    print("\nDone!")


def demo():
    """Run a demo with sample titles."""
    checkpoint_path = "cbow_final_with_vocab.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        return
    
    print("Running demo with sample titles...")
    predictor = load_predictor(checkpoint_path)
    
    demo_titles = [
        "Show HN: I built a new programming language",
        "Google announces breakthrough in quantum computing",
        "Why I left my $500k job at Facebook", 
        "The future of cryptocurrency and blockchain",
        "Building a startup from scratch in 2024",
        "New JavaScript framework changes everything",
        "Machine learning model predicts stock prices",
        "Open source alternative to expensive software",
        "How I learned to code in 6 months",
        "Security vulnerability found in popular library"
    ]
    
    print("\nPredictions for demo titles:")
    print("=" * 80)
    
    results = predictor.predict_batch(demo_titles)
    
    # Sort by predicted score (descending)
    results.sort(key=lambda x: x['predicted_score'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. Score: {result['predicted_score']:5.1f} | {result['title']}")
        print(f"     Coverage: {result['coverage']:5.1%} | Words: {result['words_in_vocab']}/{result['total_words']}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main() 