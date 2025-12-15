import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import KFold
from torch.amp import autocast  
from torch.optim import AdamW  #
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer
from torch.utils.data import ConcatDataset

from finetune.Regressor import Regressor
from finetune.DatasetPDTC import DatasetPDTC, DataCollatorPDTC

import matplotlib.pyplot as plt
warnings.simplefilter("ignore", category=FutureWarning)
os.environ["WANDB_MODE"] = "offline"

def calculate_regression_metrics(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    evs = 1 - np.var(y_true - y_pred) / np.var(y_true)
    medae = np.median(np.abs(y_true - y_pred))
    maxae = np.max(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    pearson_corr, _ = stats.pearsonr(y_true, y_pred)
    spearman_corr, _ = stats.spearmanr(y_true, y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    return {
        'mse': mse,
        'rmse': rmse,
        'evs': evs,
        'medae': medae,
        'maxae': maxae,
        'mape': mape,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mae': mae,
        'r2': r2
    }


def train_regression_model(model, train_dataset, val_dataset, batch_size=16, num_epochs=5, learning_rate=2e-5,
                               device='cuda', weight_decay=0.01, warmup_steps=0,
                               project_name="drug_patient_regression", run_name="run_1",
                               use_mixed_precision=True, gradient_accumulation_steps=1,
                               num_classes=4, regression_model=None,
                               output_dir="results", 
                               save_every_epoch_preds=True, 
                               save_best_preds=True,
                               csv_path = "metrics_log.csv"):
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "model": model.__class__.__name__,
        "optimizer": "AdamW",
        "mixed_precision": use_mixed_precision
    })

    model = model.to(device)

    data_collator = DataCollatorPDTC()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=data_collator, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.drug_model.named_parameters() if p.requires_grad],
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.patient_encoder.named_parameters()],
            "lr": learning_rate,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": learning_rate,
            "weight_decay": weight_decay
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    if use_mixed_precision and device != 'cpu':
        from torch.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    best_val_predictions = None
    best_val_targets = None

    best_val_loss = float('inf')
    train_targets, train_predictions = [], []

    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        step = 0

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            afm_features = batch['afm_features'].to(device)
            adj_features = batch['adj_features'].to(device)
            patient_encoding = batch['patient_encoding'].to(device)
            match regression_model:
                case None:
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
                case 'HyperAT':
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = None
                case 'PPIAT':
                    PPI_matrix = batch['ppi_matrix'].to(device)
                    patient_features = None
                case 'GeneAT':
                    PPI_matrix = None
                    patient_features = None
                case _:
                    patient_features = batch['patient_features'].to(device)
                    PPI_matrix = batch['ppi_matrix'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()

            if use_mixed_precision and device != 'cpu':
                with autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                    loss = F.mse_loss(outputs.squeeze(-1), labels)
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    afm_features=afm_features,
                    adj_features=adj_features,
                    patient_encoding=patient_encoding,
                    PPI_matrix=PPI_matrix,
                    patient_features=patient_features,
                )
                loss = F.mse_loss(outputs.squeeze(-1), labels)
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            step += 1
            train_loss += loss.item() * gradient_accumulation_steps
            train_targets.extend(labels.cpu().numpy())
            train_predictions.extend(outputs.squeeze(-1).detach().cpu().numpy())
            progress_bar.set_postfix({'loss': loss.item()})
            wandb.log({"batch_loss": loss.item()})

        train_avg_loss = train_loss / len(train_loader)
        train_metrics = calculate_regression_metrics(train_targets, train_predictions)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_avg_loss,
            "train_mse": train_metrics['mse'],
            "train_rmse": train_metrics['rmse'],
            "train_evs": train_metrics['evs'],
            "train_medae": train_metrics['medae'],
            "train_maxae": train_metrics['maxae'],
            "train_mape": train_metrics['mape'],
            "train_pearson": train_metrics['pearson'],
            "train_spearman": train_metrics['spearman'],
            "train_mae": train_metrics['mae'],
            "train_r2": train_metrics['r2']
        })
        model.eval()
        val_loss = 0
        val_targets, val_predictions = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                afm_features = batch['afm_features'].to(device)
                adj_features = batch['adj_features'].to(device)
                patient_encoding = batch['patient_encoding'].to(device)
                match regression_model:
                    case None:
                        patient_features = batch['patient_features'].to(device)
                        PPI_matrix = batch['ppi_matrix'].to(device)
                    case 'HyperAT':
                        patient_features = batch['patient_features'].to(device)
                        PPI_matrix = None
                    case 'PPIAT':
                        PPI_matrix = batch['ppi_matrix'].to(device)
                        patient_features = None
                    case 'GeneAT':
                        PPI_matrix = None
                        patient_features = None
                    case _:
                        patient_features = batch['patient_features'].to(device)
                        PPI_matrix = batch['ppi_matrix'].to(device)
                labels = batch['labels'].to(device)

                if use_mixed_precision and device != 'cpu':
                    with autocast("cuda"):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            afm_features=afm_features,
                            adj_features=adj_features,
                            patient_encoding=patient_encoding,
                            PPI_matrix=PPI_matrix,
                            patient_features=patient_features,
                        )
                        loss = F.mse_loss(outputs.squeeze(-1), labels)
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                    loss = F.mse_loss(outputs.squeeze(-1), labels)
                val_loss += loss.item()
                val_targets.extend(labels.cpu().numpy())
                val_predictions.extend(outputs.squeeze(-1).detach().cpu().numpy())

        val_avg_loss = val_loss / len(val_loader)
        val_metrics = calculate_regression_metrics(val_targets, val_predictions)
        wandb.log({
            "val_loss": val_avg_loss,
            "val_mse": val_metrics['mse'],
            "val_rmse": val_metrics['rmse'],
            "val_evs": val_metrics['evs'],
            "val_medae": val_metrics['medae'],
            "val_maxae": val_metrics['maxae'],
            "val_mape": val_metrics['mape'],
            "val_pearson": val_metrics['pearson'],
            "val_spearman": val_metrics['spearman'],
            "val_mae": val_metrics['mae'],
            "val_r2": val_metrics['r2']
        })

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Metrics:")
        print(f"Loss: {train_avg_loss:.4f}, MSE: {train_metrics['mse']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(
            f"EVS: {train_metrics['evs']:.4f}, MedAE: {train_metrics['medae']:.4f}, MaxAE: {train_metrics['maxae']:.4f}")
        print(
            f"MAPE: {train_metrics['mape']:.4f}%, Pearson: {train_metrics['pearson']:.4f}, Spearman: {train_metrics['spearman']:.4f}")
        print(f"MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")

        print(f"\nValidation Metrics:")
        print(f"Loss: {val_avg_loss:.4f}, MSE: {val_metrics['mse']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        print(f"EVS: {val_metrics['evs']:.4f}, MedAE: {val_metrics['medae']:.4f}, MaxAE: {val_metrics['maxae']:.4f}")
        print(
            f"MAPE: {val_metrics['mape']:.4f}%, Pearson: {val_metrics['pearson']:.4f}, Spearman: {val_metrics['spearman']:.4f}")
        print(f"MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

        if save_every_epoch_preds:
            preds_df = pd.DataFrame({
                'true_label': val_targets,
                'predicted_value': val_predictions
            })
            preds_save_path = os.path.join(output_dir, f'preds_{run_name}_epoch_{epoch + 1}.csv')
            preds_df.to_csv(preds_save_path, index=False)
            print(f"Epoch {epoch + 1} predictions saved to {preds_save_path}")

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            if save_best_preds:
                best_val_predictions = val_predictions[:]  
                best_val_targets = val_targets[:]  
                print("Best predictions updated.")
        if save_best_preds and best_val_predictions is not None:
            best_preds_df = pd.DataFrame({
                'true_label': best_val_targets,
                'predicted_value': best_val_predictions
            })
            best_preds_save_path = os.path.join(output_dir, f'best_preds_{run_name}.csv')
            best_preds_df.to_csv(best_preds_save_path, index=False)
            print(f"Final best predictions saved to {best_preds_save_path}")
    log_data = {
        "val_loss": val_avg_loss,
        "val_mse": val_metrics['mse'],
        "val_rmse": val_metrics['rmse'],
        "val_evs": val_metrics['evs'],
        "val_medae": val_metrics['medae'],
        "val_maxae": val_metrics['maxae'],
        "val_mape": val_metrics['mape'],
        "val_pearson": val_metrics['pearson'],
        "val_spearman": val_metrics['spearman'],
        "val_mae": val_metrics['mae'],
        "val_r2": val_metrics['r2']
    }
    file_exists = os.path.isfile(csv_path)

    df = pd.DataFrame([log_data])

    if file_exists:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"The metrics have been recorded. {csv_path}")

    wandb.finish()
    return model

def get_opt(args, attr_name, default_value):
    if args is None:
        return default_value
    val = getattr(args, attr_name, None)
    return val if val is not None else default_value

def CrossValidation(args=None):
    print("Starting the PDTCRegressionTenfold task.")
    model_path = get_opt(args, "model_path", "./model/multiSmiles_model/checkpoint-80000")
    tokenizer_path = get_opt(args, "tokenizer_path", "./model/custom_tokenizer")

    drug_file = get_opt(args, "drug_file", 'data/PDTC/PDTC_Drug.csv')
    ppi_file = get_opt(args, "sample_ppi_file", 'data/PDTC/PDTC_PPI.npy')

    patient_embed_file = get_opt(args, "sample_patient_embed_file", 'data/PDTC/PDTCSample_embeddings.npy')
    sensitivity_file = get_opt(args, "sample_sensitivity_file", 'data/PDTC/SampleResponse.csv')
    patient_feature_file = get_opt(args, "sample_patient_feature_file", 'data/PDTC/PDTCSample_laplacian.npy')
    patient_name_file = get_opt(args, "sample_patient_name_file", 'data/PDTC/PDTCSample_name.csv')

    Model_patient_embed_file = get_opt(args, "model_patient_embed_file", 'data/PDTC/PDTCModel_embeddings.npy')
    Model_sensitivity_file = get_opt(args, "model_sensitivity_file", "data/PDTC/ModelResponse.csv")
    Model_patient_feature_file = get_opt(args, "model_patient_feature_file",
                                         'data/PDTC/PDTCModel_laplacian.npy')
    Model_patient_name_file = get_opt(args, "model_patient_name_file", 'data/PDTC/PDTCModel_name.csv')

    num_classes = get_opt(args, "num_classes", 1)
    num_layer = get_opt(args, "num_layer", 2)
    num_heads = get_opt(args, "num_heads", 4)
    dropout_rate = get_opt(args, "dropout_rate", 0.0)
    batch_size = get_opt(args, "batch_size", 120)
    num_epochs = get_opt(args, "num_epochs", 150)
    learning_rate = get_opt(args, "learning_rate", 1e-4)
    weight_decay = get_opt(args, "weight_decay", 0.0001)
    project_name = get_opt(args, "project_name", "PDTCTenFold")

    regression_model = get_opt(args, "regression_model", None)

    print(f"Current configuration -> Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    print(f"Data path -> {drug_file}")

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    train_dataset = DatasetPDTC(
        drug_file=drug_file,
        patient_embed_file=patient_embed_file,
        sensitivity_file=sensitivity_file,
        ppi_file=ppi_file,
        patient_feature_file=patient_feature_file,
        patient_name_file=patient_name_file,
        tokenizer=tokenizer,
        num_classes=num_classes,
    )

    val_dataset = DatasetPDTC(
        drug_file=drug_file,
        patient_embed_file=Model_patient_embed_file,
        sensitivity_file=Model_sensitivity_file,
        ppi_file=ppi_file,
        patient_feature_file=Model_patient_feature_file,
        patient_name_file=Model_patient_name_file,
        tokenizer=tokenizer,
        num_classes=num_classes,
    )
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    full_dataset = ConcatDataset([train_dataset, val_dataset])
    full_size = len(full_dataset)

    kfold = KFold(n_splits=10, shuffle=True)

    indices = list(range(full_size))

    fold_datasets = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        fold_train_dataset = Subset(full_dataset, train_idx)
        fold_val_dataset = Subset(full_dataset, val_idx)

        fold_datasets.append({
            'train': fold_train_dataset,
            'val': fold_val_dataset
        })

        print(f"fold {fold + 1} :")
        print(f"Training set size: {len(fold_train_dataset)}")
        print(f"Validation set size: {len(fold_val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = 'PDTC_Tenfold'

    for fold, datasets in enumerate(fold_datasets):
        print(f"\nStarting training for fold {fold + 1}")

        current_run_name = f"{run_name}_fold_{fold + 1}"
        wandb.init(project=project_name, name=current_run_name, reinit=True)
        model = Regressor(
            drug_model_path=model_path,
            patient_encoding_dim=512,
            num_classes=num_classes,
            num_layer=num_layer,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            ParametersNum=893,
        ).to(device)

        trained_model = train_regression_model(
            model=model,
            train_dataset=datasets['train'],
            val_dataset=datasets['val'],
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            weight_decay=weight_decay,
            warmup_steps=int(len(datasets['train']) / batch_size * 5),
            project_name=project_name,
            run_name=current_run_name,
            use_mixed_precision=True,
            gradient_accumulation_steps=1,
            num_classes=num_classes,
            regression_model=regression_model,
            csv_path= f"./{project_name}_metrics_log.csv",
            output_dir=f"./{project_name}_results",
        )
        # torch.save(trained_model.state_dict(), f"./PDTC_model_{current_run_name}")
        wandb.finish()
#
if __name__ == "__main__":

    CrossValidation(args=None)



