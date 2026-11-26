"""Your Mixture-of-Experts transformer. Run it with ./your_program.sh <command> [options].

Each stage adds one command. Start with stage 01 (`tokenize`): read stage_descriptions/base-01-tokenize.md, uncomment
the code below and make it pass `make test STAGE=01 DIR=<your copy of this folder>`.

Suggested layout as you go (the reference solutions use it, but any layout passes the tests):
    app/tokenizer.py  stage 01        app/model.py   stages 02, 04-12, 17
    app/train.py      stages 03, 13   app/interp.py  stages 14-16
"""

import sys


def main():
    # Debug output goes to stderr, so it never confuses the tests, which read stdout.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the code below to pass the first stage.
    #
    # import torch
    # command = sys.argv[1]
    # if command == "tokenize":
    #     corpus = open(sys.argv[2], encoding="utf-8").read()
    #     text = sys.argv[3]
    #     chars = sorted(set(corpus))                 # the vocabulary: every distinct character
    #     stoi = {c: i for i, c in enumerate(chars)}
    #     ids = [stoi[c] for c in text]
    #     print(f"corpus characters: {len(corpus)}")
    #     print(f"vocab size: {len(chars)}")
    #     print(f"ids: {ids}")
    #     print(f"decoded: {''.join(chars[i] for i in ids)}")
    #     print(f"round trip ok: {''.join(chars[i] for i in ids) == text}")
    #     data = torch.tensor([stoi[c] for c in corpus])
    #     n = int(len(data) * 0.9)                    # first 90% to train, last 10% to check
    #     print(f"train / val characters: {n} / {len(data) - n}")
    #     g = torch.Generator().manual_seed(0)
    #     ix = torch.randint(n - 9, (4,), generator=g)
    #     x = torch.stack([data[i : i + 8] for i in ix])
    #     y = torch.stack([data[i + 1 : i + 9] for i in ix])
    #     print(f"batch shape: {tuple(x.shape)}")
    #     print(f"targets are the inputs shifted by one: {torch.equal(x[:, 1:], y[:, :-1])}")
    # else:
    #     raise RuntimeError(f"Unknown command {command}")


if __name__ == "__main__":
    main()
