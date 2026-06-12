#!/usr/bin/env python3
"""
HalluciGuard - Hallucination Detection Inference Pipeline
=========================================================
Production-quality inference script for detecting hallucinations in QA pairs.
"""

import sys
import re
import joblib
import numpy as np


MODEL_PATH = "models/answer_only_model.pkl"
VECTORIZER_PATH = "models/answer_only_vectorizer.pkl"


def clean_text(text: str) -> str:
    """Preprocess text exactly as training pipeline."""
    text = str(text)
    text = re.sub(r'\bKnowledge:\s*', '', text)
    text = re.sub(r'\bQuestion:\s*', '', text)
    text = re.sub(r'\bAnswer:\s*', '', text)
    return text.strip()


def load_artifacts():
    """Load model and vectorizer with error handling."""
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please ensure answer_only_model.pkl exists in the models/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print(f"Error: Vectorizer file not found at '{VECTORIZER_PATH}'")
        print("Please ensure answer_only_vectorizer.pkl exists in the models/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        sys.exit(1)

    return clf, vectorizer


def predict(clf, vectorizer, text: str):
    """Run inference on a single text input."""
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    
    y_pred = clf.predict(X)[0]
    y_proba = clf.predict_proba(X)[0]
    
    prob_factual = float(y_proba[0])
    prob_hallucination = float(y_proba[1])
    confidence = max(prob_factual, prob_hallucination)
    
    label = "hallucination" if y_pred == 1 else "factual"
    
    return {
        "label": label,
        "confidence": confidence,
        "prob_factual": prob_factual,
        "prob_hallucination": prob_hallucination,
        "cleaned_text": cleaned
    }


def format_output(result: dict) -> str:
    """Format prediction results for display."""
    lines = []
    lines.append("=" * 50)
    lines.append("HALLUCIGUARD PREDICTION RESULT")
    lines.append("=" * 50)
    lines.append(f"Prediction:      {result['label'].upper()}")
    lines.append(f"Confidence:      {result['confidence']:.4f}")
    lines.append(f"Probability Factual:       {result['prob_factual']:.4f}")
    lines.append(f"Probability Hallucination: {result['prob_hallucination']:.4f}")
    lines.append("-" * 50)
    
    if result['label'] == 'hallucination':
        lines.append("[!] WARNING: Potential hallucination detected!")
    else:
        lines.append("[+] Response appears factual.")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def print_usage():
    """Print usage instructions."""
    print("""
HalluciGuard - Hallucination Detection
======================================

Usage:
  python predict.py                    # Interactive mode
  python predict.py "text to analyze"  # Single prediction
  python predict.py --help             # Show this help

Interactive Mode:
  Enter text when prompted. Type 'quit' or 'exit' to quit.
  Empty lines are ignored.

Input Format:
  The model accepts raw text (Knowledge/Question/Answer format or plain text).
  Preprocessing removes 'Knowledge:', 'Question:', 'Answer:' prefixes automatically.

Examples:
  python predict.py "Knowledge: Paris is capital of France. Question: What is capital? Answer: Paris."
  python predict.py "The moon is made of cheese."
""")


def interactive_mode(clf, vectorizer):
    """Run interactive prediction loop."""
    print("\nHalluciGuard Interactive Mode")
    print("Type 'quit' or 'exit' to exit\n")
    
    while True:
        try:
            text = input("Enter text > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if text.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        try:
            result = predict(clf, vectorizer, text)
            print(format_output(result))
            print()
        except Exception as e:
            print(f"Error during prediction: {e}\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('--help', '-h', 'help'):
        print_usage()
        return
    
    print("Loading HalluciGuard model...")
    clf, vectorizer = load_artifacts()
    print("Model loaded successfully.\n")
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        try:
            result = predict(clf, vectorizer, text)
            print(format_output(result))
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)
    else:
        interactive_mode(clf, vectorizer)


if __name__ == "__main__":
    main()