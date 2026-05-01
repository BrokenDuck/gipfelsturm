#!/usr/bin/env python3
import argparse
import math


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tok-s-gpu", type=float, required=True)
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--global-batch", type=int, required=True)
    p.add_argument("--minutes", type=float, required=True)
    args = p.parse_args()

    total_tok_s = args.tok_s_gpu * args.gpus
    target_tokens = total_tok_s * args.minutes * 60
    tokens_per_step = args.seq_len * args.global_batch
    steps = math.floor(target_tokens / tokens_per_step)

    print(f"total_tok_s:       {total_tok_s:,.0f}")
    print(f"target_tokens:     {target_tokens:,.0f}")
    print(f"tokens_per_step:   {tokens_per_step:,.0f}")
    print(f"target_steps:      {steps:,}")
    print(f"estimated_minutes: {steps * tokens_per_step / total_tok_s / 60:.2f}")


if __name__ == "__main__":
    main()
