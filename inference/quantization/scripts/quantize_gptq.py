import argparse
import importlib
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize model with GPTQConfig")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dataset", default="c4")
    parser.add_argument("--desc-act", action="store_true")
    args = parser.parse_args()

    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    GPTQConfig = getattr(transformers, "GPTQConfig")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    gptq_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        dataset=args.dataset,
        desc_act=args.desc_act,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=gptq_config,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = {
        "method": "gptq",
        "model_id": args.model_id,
        "output_dir": args.output_dir,
        "bits": args.bits,
        "group_size": args.group_size,
        "dataset": args.dataset,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
