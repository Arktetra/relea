"""
Script for training a BPE tokenizer.
"""

from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from pathlib import Path

import argparse

def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

def main(args):

    dataset = load_dataset(args.data_dir, name=args.data_file, split="train")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=1024, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path), pretty=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tokenizer Trainer")
    parser.add_argument(
        "--data-dir", 
        default="wikitext", 
        type=str,
        help="The data directory where the dataset is located.",
    )
    parser.add_argument(
        "--data-file",
        default="wikitext-2-raw-v1",
        type=str,
        help="The data file on which the tokenizer will be trained on."
    )
    parser.add_argument(
        "--vocab-size",
        default=1024,
        type=int,
        help="The size of the vocabulary to train for."
    )
    parser.add_argument(
        "--save-path",
        default="tokenizer.json",
        type=str,
        help="The path where the tokenizer will be saved."
    )
    args = parser.parse_args()

    main(args)