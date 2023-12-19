
# PreIS: A Novel Data Augmentation Approach Using Protein Language Models for Influenza A Subtype Prediction

## Description

Protein sequences have complex structures and are subject to strict rules regarding their composition and function; therefore, it is difficult to generate new examples of protein sequences without violating these rules or introducing errors that could affect the accuracy of the model. To overcome this problem, it is important to develop approaches for data augmentation that are tailored specifically to the characteristics and constraints of protein sequences. In this paper, we propose a novel approach for producing synthetic samples of protein sequences that leverages supervision to preserve data distribution and improve the generalization capabilities of the model. Our approach involves the use of labeled data to guide the generation of augmented samples, ensuring that the resulting synthetic data is more representative of the distribution of the actual data. 

![SDA](https://github.com/mohaminn/sda/blob/main/Images/fig_sda.jpeg)

The implemented MC-NN model is available [here](https://github.com/CBRC-lab/PreIS/tree/main/MC_NN).

## Installation

This project is available via huggingface/transformers:

```bash
pip install torch
pip install transformers
```
    
## Usage

```bash
cd PreIS
python3 train.py
```

