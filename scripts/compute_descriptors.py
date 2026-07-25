import argparse

from router.descriptors import save_descriptors
from router.utils import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Compute logit-footprint descriptor")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local dir")
    parser.add_argument("--probes_files", required=True, help="Path to JSON files with probe prompts", nargs="+")
    parser.add_argument("--out", required=True, help="Path to .npy file to save descriptor")
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--n_tokens", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--four_bit", action="store_true",
                        help="Load the model with 4-bit NF4 quantization (bitsandbytes) to fit tighter VRAM")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, four_bit=args.four_bit)

    save_descriptors(
        model=model,
        tokenizer=tokenizer,
        probes=args.probes_files,
        out=args.out,
        topk=args.topk,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
    )
    

if __name__ == "__main__":
    main()
