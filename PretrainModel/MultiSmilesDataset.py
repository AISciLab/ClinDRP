import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import json
import ast
import os
from tokenizers import ByteLevelBPETokenizer

from PretrainModel.MultiSmilesModel import MultiSmilesModelConfig, MultiSmilesModelForMaskedLM


class MultiSmilesDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']

        afm_matrix = self._process_matrix(row['afm'])

        adj_matrix = self._process_matrix(row['adj'])

        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'afm_features': torch.tensor(afm_matrix, dtype=torch.float32),
            'adj_features': torch.tensor(adj_matrix, dtype=torch.float32),
        }

        return item

    def _process_matrix(self, matrix_str):
        try:
            matrix = ast.literal_eval(matrix_str)
            return np.array(matrix, dtype=np.float32)
        except (SyntaxError, ValueError):
            try:
                matrix = json.loads(matrix_str)
                return np.array(matrix, dtype=np.float32)
            except json.JSONDecodeError:
                print(f"Unable to parse the matrix string: {matrix_str[:50]}...")
                feature_dim = 27 if 'afm' in matrix_str else 3
                return np.zeros((1, feature_dim), dtype=np.float32)


class MultimodalDataCollatorForMLM(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch_size = len(features)
        if batch_size == 0:
            return {}

        input_ids = torch.stack([f.pop("input_ids") for f in features])
        attention_mask = torch.stack([f.pop("attention_mask") for f in features])
        afm_features = [f.pop("afm_features") for f in features]
        adj_features = [f.pop("adj_features") for f in features]

        max_afm_len = max(af.shape[0] for af in afm_features)
        padded_afm = []
        for af in afm_features:
            if af.shape[0] < max_afm_len:
                padding = torch.zeros((max_afm_len - af.shape[0], af.shape[1]), dtype=af.dtype, device=af.device)
                padded_afm.append(torch.cat([af, padding], dim=0))
            else:
                padded_afm.append(af)

        max_adj_len = max(adj.shape[0] for adj in adj_features)
        padded_adj = []
        for adj in adj_features:
            if adj.shape[0] < max_adj_len:
                padding = torch.zeros((max_adj_len - adj.shape[0], adj.shape[1]), dtype=adj.dtype, device=adj.device)
                padded_adj.append(torch.cat([adj, padding], dim=0))
            else:
                padded_adj.append(adj)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "afm_features": torch.stack(padded_afm),
            "adj_features": torch.stack(padded_adj),
        }

        if self.mlm:
            inputs, labels = self.torch_mask_tokens(batch["input_ids"].clone())
            batch["input_ids"] = inputs
            batch["labels"] = labels

        return batch

def initialize_model(tokenizer):
    config = MultiSmilesModelConfig.from_pretrained(
        'roberta-base',
        afm_dim=27,
        adj_dim=3,  
        vocab_size=len(tokenizer),
        num_attention_heads=12,
        num_hidden_layers=6,
    )
    model = MultiSmilesModelForMaskedLM(config)
    return model

def train_new_tokenizer(train_files=None, output_dir="./custom_tokenizer"):
    if train_files is None:
        df = pd.read_csv(train_files)
        smiles_list = df['smiles'].tolist()

        os.makedirs("temp", exist_ok=True)
        train_file = "temp/smiles_corpus.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")

        train_files = [train_file]

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=train_files,
        vocab_size=512, 
        min_frequency=3, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)

    fast_tokenizer = RobertaTokenizerFast.from_pretrained(output_dir)
    fast_tokenizer.mask_token = "<mask>"

    return fast_tokenizer


def train_model(model, train_dataset, eval_dataset=None, tokenizer=None, output_dir="model/smilesOnly_roberta_model"):

    data_collator = MultimodalDataCollatorForMLM(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=128,
        save_steps=20_000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        logging_dir='./logs',
        logging_steps=500,
        save_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    print("Save the model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    print(f"The model has been saved to {output_dir}")

    return model, trainer

def get_opt(args, attr_name, default_value):
    if args is None:
        return default_value
    val = getattr(args, attr_name, None)
    return val if val is not None else default_value

def Pretrain(args):
    train_data = get_opt(args, "train_data", 'data/Pretrain/out_embedding_train.csv')
    eval_data = get_opt(args, "eval_data", ".data/Pretrain/out_embedding_eval.csv")
    Pretrain_output_dir = get_opt(args, "Pretrain_output_dir", "model/Pretrain_model")

    print("Starting to train the custom tokenizer...")
    tokenizer = train_new_tokenizer(
        train_files= None, 
        output_dir="model/Mulit_roberta_tokenizer"
    )
    print("The tokenizer training is complete.")

    train_dataset = MultiSmilesDataset(
        csv_path=train_data,
        tokenizer=tokenizer,
        max_length=512,
    )

    eval_dataset = MultiSmilesDataset(
        csv_path=eval_data,
        tokenizer=tokenizer,
        max_length=512,
    )

    model = initialize_model(tokenizer)

    model, trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir = Pretrain_output_dir
    )

    print("Model training complete!")


if __name__ == "__main__":

    Pretrain(args)

