import hydra
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    dataset_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    hex_path = hydra.utils.to_absolute_path(cfg.tokenizers.hex_path)
    c_path = hydra.utils.to_absolute_path(cfg.tokenizers.c_path)

    df = pd.read_parquet(dataset_path)

    hex_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    hex_tokenizer.pre_tokenizer = Whitespace()

    hex_trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    hex_tokenizer.train_from_iterator(df["hex"].tolist(), trainer=hex_trainer)
    hex_tokenizer.save(hex_path)

    c_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    c_tokenizer.pre_tokenizer = Whitespace()

    c_trainer = BpeTrainer(
        vocab_size=8000, special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    c_tokenizer.train_from_iterator(df["target"].tolist(), trainer=c_trainer)
    c_tokenizer.save(c_path)
    print("Tokenizers trained and saved successfully!")


if __name__ == "__main__":
    main()
