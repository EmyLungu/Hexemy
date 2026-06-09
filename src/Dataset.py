import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence


class HexDataset(IterableDataset):
    def __init__(self, hf_dataset, hex_tokenizer, c_tokenizer):
        self.hf_dataset = hf_dataset
        self.hex_tokenizer = hex_tokenizer
        self.c_tokenizer = c_tokenizer

        self.hex_tokenizer.enable_truncation(max_length=2048)
        self.c_tokenizer.enable_truncation(max_length=2048)

    def __iter__(self):
        for row in self.hf_dataset:
            raw_hex = row["asm"]
            raw_c = row["code"]

            hex_tokens = self.hex_tokenizer.encode(raw_hex).ids
            c_tokens = self.c_tokenizer.encode(raw_c).ids

            src_tensor = torch.tensor(hex_tokens, dtype=torch.long)
            tgt_tensor = torch.tensor(c_tokens, dtype=torch.long)

            yield src_tensor, tgt_tensor


def collate_fn(batch):
    """
    Takes a list of tuples (src_tensor, tgt_tensor) from the Dataset
    and pads them so they can be stacked into unified batches.
    """
    src_list, tgt_list = zip(*batch)

    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return src_padded, tgt_padded
