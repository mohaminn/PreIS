import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import utils
import config


def train_fn(
    model, 
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples,
    acc_steps, 
    augment_ids,
    augment_targets,
):

    model = model.train()
    
    losses= []
    correct_predictions = 0 
    
    for b_idx, d in tqdm(enumerate(data_loader), total = len(data_loader)):

        input_ids = d['ids'].to(device)
        attention_mask = d['mask'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        targets = d['targets'].to(device, dtype = torch.long)
        
        if (config.SDA):
            input_ids = utils.SDA(input_ids, targets, augment_ids, augment_targets)
        
        outputs = model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)
        loss = loss / acc_steps
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), max_norm = 1.0)
        if (((b_idx + 1) % acc_steps == 0) or (b_idx + 1 == len(data_loader))):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
    return correct_predictions.double()/n_examples, np.mean(losses)



def eval_fn(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0
  
    with torch.no_grad():
  
        for _, d in tqdm(enumerate(data_loader), total = len(data_loader)):
  
            input_ids = d['ids'].to(device)
            attention_mask = d['mask'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            targets = d['targets'].to(device, dtype = torch.long)
            outputs = model(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
        
    return correct_predictions.double() / n_examples , np.mean(losses)






