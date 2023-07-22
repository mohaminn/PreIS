from transformers import AutoTokenizer
import torch 
import config


class dataset: 
    def __init__(self, sequence, target):
        self.sequence = sequence
        self.target = target
        self.maxLength = config.MAX_LEN
        self.tokenizer = config.tokenizer

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        sequence = self.sequence[item]

        inputs = self.tokenizer.encode_plus(
            sequence,
            None,
            add_special_tokens = True,
            max_length = 1024,
            padding = "max_length",
            truncation = True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {
            "ids": torch.tensor(ids, dtype= torch.long),
            "mask": torch.tensor(mask, dtype= torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype= torch.long),
            "targets" : torch.tensor(self.target[item], dtype= torch.float)
        }