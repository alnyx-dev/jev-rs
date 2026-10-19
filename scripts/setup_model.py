#!/usr/bin/env python3
"""
Model preparation script for Jev-RS.
Exports BGE-M3 Zeroshot to ONNX format.
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Export a HuggingFace NLI model to ONNX for Jev-RS")
    parser.add_argument("--model", default="MoritzLaurer/bge-m3-zeroshot-v2.0", help="HuggingFace model ID")
    parser.add_argument("--out", default="model_onnx", help="Output directory path")
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("[!] Error: optimum and transformers are required.")
        print("[!] Run: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    print(f"[*] Downloading and exporting '{args.model}'...")
    os.makedirs(args.out, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[+] Model and tokenizer saved to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()