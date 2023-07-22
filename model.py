import torch 
from torch import nn
import config
import rita_modeling
import utils



class RITA_s(nn.Module):
    def __init__(self):
        super(RITA_s, self).__init__()
        self.rita = utils.init_rita()
        self.cls_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 7)
        )
        
    def pooler_fn(
        self, 
        token_embeddings, 
        attention_mask,
    ): 

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)        
        output_vector = sum_embeddings / sum_mask

        return output_vector      

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        
        x = self.rita(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        x = self.pooler_fn(token_embeddings = x[0], attention_mask= attention_mask)
        x = self.cls_head(x)

        return x 




