from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import torch
from torch.utils.data import Dataset
import argparse
import json

class LMDataset(Dataset):
    """
    A PyTorch Dataset for language modeling.
    Concatenates all text, tokenizes, and returns samples of length (context_length + 1).
    Each sample is a tensor of token IDs.
    """
    def __init__(self, file_path, tokenizer_dir, context_length=512):

        if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
            raise ValueError(f"Tokenizer not found in {tokenizer_dir}. Please train or provide a valid tokenizer.")
        else:
            self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.context_length = context_length

        # Read and concatenate all text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the entire corpus at once
        ids = self.tokenizer.encode(text).ids

        # Split into samples of length context_length + 1
        self.samples = []
        for i in range(0, len(ids) - context_length, context_length):
            chunk = ids[i : i + context_length + 1]
            if len(chunk) == context_length + 1:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_tokenizer(file_path, vocab_size=32000, save_dir="data/tokenizer"):
    """Trains a BPE tokenizer on the given dataset."""
    save_path = os.path.join(save_dir, "tokenizer.json")
    if os.path.exists(save_path):
        print("Tokenizer: Loading existing tokenizer from:", save_path)
        tokenizer = Tokenizer.from_file(save_path)
        return tokenizer
    
    print("Tokenizer: Training new tokenizer and saving to:", save_path)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
    tokenizer.train([file_path], trainer)
    print("Tokenizer: Trained tokenizer vocab size:", tokenizer.get_vocab_size())
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tokenizer.save(save_path)
    
    # Prepare tokenizer metadata
    tokenizer_metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "unk_token_id": tokenizer.token_to_id("<unk>"),
        "bos_token_id": tokenizer.token_to_id("<s>"),
        "eos_token_id": tokenizer.token_to_id("</s>")
    }
    print("Tokenizer: Trained tokenizer vocab size:", tokenizer.get_vocab_size())
    train_tokenizer(args.file_path, args.vocab_size, args.save_dir)
    metadata_path = os.path.join(save_dir, "tokenizer_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_metadata, f, ensure_ascii=False, indent=2)
    print("Tokenizer: Saved tokenizer metadata to:", metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("file_path", type=str, help="Path to the training text file")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size (default: 32000)")
    parser.add_argument("--save_dir", type=str, default="data/tokenizer", help="Directory to save tokenizer")
    
    args = parser.parse_args()
    
    train_tokenizer(args.file_path, args.vocab_size, args.save_dir)