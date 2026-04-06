import argparse
import importlib
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Load AWQ quantized model with runtime options")
    parser.add_argument("--model-id", required=True, help="AWQ quantized model id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="gemm")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--do-fuse", action="store_true")
    parser.add_argument("--fuse-max-seq-len", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="")
    args = parser.parse_args()

    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AwqConfig = getattr(transformers, "AwqConfig")

    if args.do_fuse and args.attn_implementation == "flash_attention_2":
        raise ValueError("AWQ fused modules should not be combined with FlashAttention2")

    quant_config = AwqConfig(
        bits=args.bits,
        version=args.version,
        do_fuse=args.do_fuse,
        fuse_max_seq_len=args.fuse_max_seq_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    kwargs = {
        "quantization_config": quant_config,
        "device_map": "auto",
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = {
        "method": "awq",
        "model_id": args.model_id,
        "output_dir": args.output_dir,
        "version": args.version,
        "bits": args.bits,
        "do_fuse": args.do_fuse,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
