import config
import sys
sys.path.append(config.RITA_DIR)
import dataset
import engine
import utils
import torch
from torch import nn
import pandas as pd
import numpy as np
from model import RITA_s
from sklearn import model_selection, metrics
from transformers import get_linear_schedule_with_warmup, AdamW
from collections import defaultdict

print ("Imports are done...")

def run():

    utils.random_seed()
    
    df_train = pd.read_csv(config.TRAIN_DATA_PATH).fillna("none")
    df_valid = pd.read_csv(config.VLID_DATA_PATH).fillna("none")
    df_test =  pd.read_csv(config.TEST_DATA_PATH).fillna("none")

    train_dataset = dataset.dataset(
        sequence=df_train.sequence.values, target=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle = True, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2
    )

    valid_dataset = dataset.dataset(
        sequence=df_valid.sequence.values, target=df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    test_dataset = dataset.dataset(
        sequence=df_test.sequence.values, target=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = RITA_s()
    model.to(device)

    temp = config.tokenizer.batch_encode_plus(
                list(df_train.sequence),
                add_special_tokens = True,
                max_length = 1024,
                padding = "max_length",
                truncation = True,
                return_tensors = "pt"
            )

    sda_data_ids = temp["input_ids"].to(device)
    sda_data_targets = torch.tensor(df_train.label.values).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    num_train_steps = int(len(df_train) / (config.TRAIN_BATCH_SIZE * config.ACCUMULATION_STEPS) * (config.EPOCHS + 30) * 2)
    optimizer = AdamW(optimizer_parameters, lr=2e-5, eps=1e-8, no_deprecation_warning = True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=num_train_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)
    
    history = defaultdict(list)
    best_accuracy = 0

    if(not config.FINE_TUNNING):
        for param in model.rita.parameters():
            param.requires_grad = False

    
    for epoch in range(config.EPOCHS):  

        train_acc, train_loss = engine.train_fn(
        model, 
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train),
        config.ACCUMULATION_STEPS,
        sda_data_ids,
        sda_data_targets
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = engine.eval_fn(
            model, 
            valid_data_loader,
            loss_fn,
            device,
            len(df_valid)
        )
            
        print(f'Val loss {val_loss} val accuracy {val_acc}')
        print()
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if val_acc >= best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

    model.load_state_dict(torch.load('best_model_state.bin'))
    test_acc, test_loss = engine.eval_fn(
        model, 
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )        
    print(f'Test loss {test_loss} test accuracy {test_acc}')
    print()

    return history

if __name__ == "__main__":
    run()