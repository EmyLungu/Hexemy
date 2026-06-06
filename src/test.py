import hydra
from omegaconf import DictConfig
import torch
from tokenizers import Tokenizer
from Model import HexModel


def decompile(
    raw_hex_string, model, hex_tokenizer, c_tokenizer, device, max_length=128
):
    hex_pad_idx = hex_tokenizer.token_to_id("[PAD]")
    c_pad_idx = c_tokenizer.token_to_id("[PAD]")
    sos_idx = c_tokenizer.token_to_id("[SOS]")
    eos_idx = c_tokenizer.token_to_id("[EOS]")

    hex_tokenizer.enable_truncation(max_length=2048)
    hex_tokens = hex_tokenizer.encode(raw_hex_string).ids

    src_tensor = torch.tensor([hex_tokens], dtype=torch.long).to(device)
    tgt_tokens = [sos_idx]

    temperature = 0.6
    top_p = 0.9
    repetition_penalty = 1.5

    print("Generating code token-by-token...", flush=True)
    with torch.no_grad():
        for i in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(
                device
            )

            output = model(src_tensor, tgt_tensor, hex_pad_idx, c_pad_idx)
            next_token_logits = output[0, -1, :]

            for token_id in set(tgt_tokens):
                if token_id in [sos_idx, c_pad_idx]:
                    continue

                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

            next_token_logits /= temperature

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("Inf")

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(
                probabilities, num_samples=1
            ).item()

            tgt_tokens.append(next_token_id)

            if (next_token_id) == eos_idx:
                break

    return c_tokenizer.decode(tgt_tokens)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hex_tokenizer_path = hydra.utils.to_absolute_path(cfg.tokenizers.hex_path)
    c_tokenizer_path = hydra.utils.to_absolute_path(cfg.tokenizers.c_path)

    checkpoint_dir = hydra.utils.to_absolute_path(cfg.checkpoint.dir)
    checkpoint_path = f"{checkpoint_dir}/{cfg.checkpoint.filename}"

    device = torch.device(
        "cuda"
        if (cfg.training.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    hex_tokenizer = Tokenizer.from_file(hex_tokenizer_path)
    c_tokenizer = Tokenizer.from_file(c_tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    HEX_VOCAB_SIZE = checkpoint["hex_vocab_size"]
    C_VOCAB_SIZE = checkpoint["c_vocab_size"]

    model = HexModel(
        hex_vocab_size=HEX_VOCAB_SIZE, c_vocab_size=C_VOCAB_SIZE
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_hex = "55 48 89 e5 48 83 ec 10 c7 45 fc 00 00 00 00 8b 45 fc 5d c3"

    print("\n--- TARGET MACHINE HEX ---")
    print(sample_hex)

    decompiled_c = decompile(
        sample_hex, model, hex_tokenizer, c_tokenizer, device
    )
    print("\n--- MODEL PREDICTED C CODE ---")
    print(decompiled_c)


if __name__ == '__main__':
    main()
