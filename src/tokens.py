import hydra
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset


def batch_iterator(dataset, column_name):
    """A generator that yields text from a specific column
    in a streaming dataset."""
    for item in dataset:
        yield item[column_name]


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hex_path = hydra.utils.to_absolute_path(cfg.tokenizers.hex_path)
    c_path = hydra.utils.to_absolute_path(cfg.tokenizers.c_path)

    ds = load_dataset(
        "LLM4Binary/decompile-bench", split="train", streaming=True
    )

    hex_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    hex_tokenizer.pre_tokenizer = Whitespace()

    hex_trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    hex_tokenizer.train_from_iterator(
        batch_iterator(ds, "asm"), trainer=hex_trainer
    )
    hex_tokenizer.save(hex_path)

    c_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    c_tokenizer.pre_tokenizer = Whitespace()

    c_trainer = BpeTrainer(
        vocab_size=8000, special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    c_tokenizer.train_from_iterator(
        batch_iterator(ds, "code"), trainer=c_trainer
    )
    c_tokenizer.save(c_path)
    print("Tokenizers trained and saved successfully!")


if __name__ == "__main__":
    main()
