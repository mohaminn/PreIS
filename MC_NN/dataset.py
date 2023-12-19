from transformers import AutoTokenizer
import torch 
import config
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class dataset: 

    def __init__(self, sequence, target):
        self.sequence = sequence
        self.target = target
        self.maxLength = config.MAX_LEN
        self.tokenizer = config.tokenizer
        self.onehot_encoder, self.label_encoder = self.fit_encoders()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        sequence = self.sequence[item]

        trigram = self.generate_N_grams(sequence)
        one_hot = self.one_hot_encoder(trigram)

        return {
            "inputs": torch.tensor(one_hot, dtype= torch.float),
            "targets" : torch.tensor(self.target[item], dtype= torch.float)
        }

    def generate_N_grams(self, text,ngram=3):
  
        words=[word for word in (" ".join(text)).split(" ")]  
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=["".join(ngram) for ngram in temp]

        return ans

    def one_hot_encoder(self, item):

        integer_encoded = self.label_encoder.transform(self.generate_N_grams(item))
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = self.onehot_encoder.transform(integer_encoded)

        return onehot_encoded

    def fit_encoders(self):

        aminos = ['Y', 'H', 'L', 'F', 'K', 'X', 'G', 'Z', 'I', 'Q', 'C', 'M', 'N', 'A', 'V', 'J', 'T', 'P', 'W', 'D', 'S', 'E', 'B', 'R']
        dic = []

        for i1 in aminos:
          for i2 in aminos:
            for i3 in aminos:
              dic.append(i1 + i2 + i3)
        values = np.array(dic)
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(values)
        integer_encoded = label_encoder.transform(values)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = OneHotEncoder(sparse_output = False)
        onehot_encoder = onehot_encoder.fit(integer_encoded)
        
        return (onehot_encoder, label_encoder)






