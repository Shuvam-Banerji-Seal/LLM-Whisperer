import argparse
import importlib
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize model with bitsandbytes")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["int8", "nf4"], default="nf4")
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--double-quant", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    BitsAndBytesConfig = getattr(transformers, "BitsAndBytesConfig")

    compute_dtype = getattr(torch, args.compute_dtype)

    if args.mode == "int8":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=args.cpu_offload,
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = {
        "method": "bitsandbytes",
        "mode": args.mode,
        "model_id": args.model_id,
        "output_dir": args.output_dir,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
