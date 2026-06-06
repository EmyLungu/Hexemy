import os
import time

import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from Model import HexModel
from Dataset import HexDataset, collate_fn


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hex_tokenizer_path = hydra.utils.to_absolute_path(cfg.tokenizers.hex_path)
    c_tokenizer_path = hydra.utils.to_absolute_path(cfg.tokenizers.c_path)
    dataset_path = hydra.utils.to_absolute_path(cfg.dataset.path)

    hex_tokenizer = Tokenizer.from_file(hex_tokenizer_path)
    c_tokenizer = Tokenizer.from_file(c_tokenizer_path)

    hex_vobcab_size = hex_tokenizer.get_vocab_size()
    c_vocab_size = c_tokenizer.get_vocab_size()
    hex_pad_idx = hex_tokenizer.token_to_id("[PAD]")
    c_pad_idx = c_tokenizer.token_to_id("[PAD]")

    accumulation_steps = (
        cfg.training.target_effective_batch // cfg.training.batch_size
    )
    device = torch.device(
        "cuda"
        if (cfg.training.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    dataset = HexDataset(dataset_path, hex_tokenizer, c_tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = HexModel(
        hex_vocab_size=hex_vobcab_size, c_vocab_size=c_vocab_size
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=c_pad_idx)

    print(f"Total rows in DataFrame: {len(dataset.df)}")
    print(f"Total samples in Dataset: {len(dataset)}")
    print(f"Total batches in DataLoader: {len(dataloader)}")

    print("Started Training")

    start_time = time.time()

    model.train()
    for epoch in range(cfg.training.epochs):
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            output = model(src, tgt_input, hex_pad_idx, c_pad_idx)

            loss = criterion(
                output.reshape(-1, c_vocab_size), tgt_expected.reshape(-1)
            )
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(dataloader)

        elapsed_time = int(time.time() - start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        print(
            f"Epoch [{epoch+1}/{cfg.training.epochs}] | "
            f"Average Loss: {avg_loss:.4f} "
            f"| ({minutes:02d}min:{seconds:02d}s)"
        )

    elapsed_time = int(time.time() - start_time)
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"Total training time: {minutes:02d}min:{seconds:02d}s")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hex_vocab_size": hex_vobcab_size,
        "c_vocab_size": c_vocab_size,
    }

    checkpoint_dir = hydra.utils.to_absolute_path(cfg.checkpoint.dir)
    checkpoint_path = f"{checkpoint_dir}/{cfg.checkpoint.filename}"
    torch.save(checkpoint, checkpoint_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
