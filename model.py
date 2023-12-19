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

class MC_NN(nn.Module):
    def __init__(self):
        super(MC_NN, self).__init__()
        self.embedding = nn.Linear(13824, 128 * 5)
        self.transformer = nn.TransformerEncoderLayer(d_model=128*5, nhead=5)
        self.cls_head = nn.Sequential(
            nn.Linear(128*5, 64 *4),
            nn.ReLU(),
            nn.Linear(64 *4, 32 *4),
            nn.ReLU(),
            nn.Linear(32 *4, 16 *4),
            nn.ReLU(),
            nn.Linear(16 *4, 7)
        )
        self.p = self.getPositionEncoding(seq_len = 600)

    def getPositionEncoding(self, seq_len, d = 128 * 5, n=10000):
        P = torch.zeros((seq_len, d)).to(torch.device("cuda"))
        for k in range(seq_len):
            for i in torch.arange(int(d/2)):
                denominator = n ** (2*i/d)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P.detach()


    def pooler_fn(
        self, 
        token_embeddings, 
    ): 

        sum_embeddings = torch.sum(token_embeddings, 1)
        sum_mask = token_embeddings.size(1)
        output_vector = sum_embeddings / sum_mask

        return output_vector      

    def forward(
        self,
        inputs,
    ):
        p_e = self.p[:inputs.size(1), :]
        x = self.embedding(inputs) + p_e
        x = self.transformer(x)
        x = self.pooler_fn(token_embeddings = x)
        x = self.cls_head(x)

        return x 



