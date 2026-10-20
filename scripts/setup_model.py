#!/usr/bin/env python3
"""
Optional: export a custom HuggingFace model to ONNX for Jev-RS.

The reference model ships a pre-built ONNX export, so the normal setup needs
no Python at all:

    cargo build --release
    ./target/release/jev-rs --download-model

This script is only needed for a model that does NOT publish an ONNX export.
It converts the model and validates the export against Jev-RS's
single-forward-pass expectations (input_ids, attention_mask inputs;
classification logits output).

    pip install optimum[onnxruntime] transformers torch
    python scripts/setup_model.py --model <repo-id> --out model_onnx
"""
import argparse
import os
import sys
from importlib.metadata import version as pkg_version


def main():
    parser = argparse.ArgumentParser(description="Export a HuggingFace NLI model to ONNX for Jev-RS")
    parser.add_argument("--model", default="MoritzLaurer/bge-m3-zeroshot-v2.0", help="HuggingFace model ID")
    parser.add_argument("--out", default="model_onnx", help="Output directory path")
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError:
        print("[!] Error: optimum (with the onnxruntime extra) is required.")
        print("[!] Run: pip install optimum[onnxruntime]")
        sys.exit(1)

    try:
        import torch  # noqa: F401 — required by the optimum export path
    except ImportError:
        print("[!] Error: torch is required for the ONNX export.")
        print("[!] Run: pip install torch")
        sys.exit(1)

    from transformers import AutoTokenizer

    for pkg in ("optimum", "transformers", "torch"):
        try:
            print(f"[*] {pkg} version: {pkg_version(pkg)}")
        except Exception:
            pass

    print(f"[*] Downloading and exporting '{args.model}'...")
    os.makedirs(args.out, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[+] Model and tokenizer saved to: {os.path.abspath(args.out)}")

    # Validate the export against Jev-RS expectations: the server runs a single
    # forward pass over [input_ids, attention_mask] and reads the entailment
    # logit from the classification head.
    onnx_path = os.path.join(args.out, "model.onnx")
    if not os.path.exists(onnx_path):
        print("[!] Export validation skipped: model.onnx not found in output dir.")
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] onnxruntime not importable, skipping ONNX IO validation.")
        return

    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inames = sorted(i.name for i in sess.get_inputs())
        onames = sorted(o.name for o in sess.get_outputs())
        print(f"[+] ONNX IO check: inputs={inames}, outputs={onames}")
        if "input_ids" not in inames or "attention_mask" not in inames:
            print("[!] WARNING: expected 'input_ids' and 'attention_mask' inputs.")
            sys.exit(1)
    except Exception as e:
        print(f"[!] Export validation failed: {e}")
        sys.exit(1)

    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if id2label:
        print(f"[+] id2label: {id2label}")
        print("[*] The Rust server resolves the entailment index from config.json (id2label).")


if __name__ == "__main__":
    main()
