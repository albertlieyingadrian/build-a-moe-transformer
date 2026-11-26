In this stage, you'll implement a `tokenize` command that turns text into numbers and back, splits it into training
and validation text, and cuts it into batches.

### Characters to numbers

A model only sees numbers. The simplest tokenizer gives every distinct character in the corpus an id: sort the
characters, and a character's id is its place in the list. Tiny Shakespeare has 65 of them (letters, digits,
punctuation, space and newline).

```
chars = sorted(set(corpus))        ['\n', ' ', '!', ..., 'a', 'b', ..., 'z']
encode("To be") = [32, 53, 1, 40, 43]
decode([32, 53, 1, 40, 43]) = "To be"
```

Real models use subword tokenizers (DeepSeek-V3 has 129k tokens), but characters keep the vocabulary small and let
every later stage run on a laptop. Nothing in the MoE part of the course depends on the choice.

### Two piles and a sliding window

Keep the last 10% of the text aside as **validation** text. The model never trains on it, so its loss there tells you
whether it learned the language or memorised the page.

A training **batch** is a few random windows of `block_size` characters. The target for each position is simply the
next character, so the targets are the same window shifted one to the right:

```
x = "First Ci"     y = "irst Cit"
```

### Worked example

```bash
$ ./your_program.sh tokenize data/tinyshakespeare.txt "To be"
corpus characters: 1115394
vocab size: 65
ids: [32, 53, 1, 40, 43]
decoded: To be
round trip ok: True
train / val characters: 1003854 / 111540
batch shape: (4, 8)
targets are the inputs shifted by one: True
```

`make data` downloads the 1 MB Tiny Shakespeare file into `data/`.

### Tests

The tests run `tokenize` on a small corpus and check the vocabulary, the ids, the round trip, a 90/10 split, a
`(4, 8)` batch and the one-character shift.

### Check yourself

1. Why split by position (the last 10%) rather than picking random characters for validation?
2. What would go wrong if `y` were the same window as `x`, not shifted?
