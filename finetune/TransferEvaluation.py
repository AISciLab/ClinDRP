import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.amp import autocast 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer
from sklearn.manifold import TSNE
from adabelief_pytorch import AdaBelief
from scipy import stats

from finetune.Classifier import Classifier
from finetune.DatasetPDTC import DatasetPDTC, DataCollatorPDTC

import matplotlib.pyplot as plt
from finetune.Analsy import extract_atom_importance_from_classifier

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


def train_classification_model(model, train_dataset, val_dataset, batch_size=16, num_epochs=5, learning_rate=2e-5,
                               device='cuda', weight_decay=0.01, warmup_steps=0,
                               project_name="drug_patient_regression", run_name="run_1",
                               use_mixed_precision=True, gradient_accumulation_steps=1,
                               num_classes=4,regression_model=None,
                               output_dir="results", 
                               save_every_epoch_preds=False, 
                               save_best_preds=False,
                               csv_path="metrics_log.csv",
                               draw_tsne=False,
                               save_features=False, save_importance=False):
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

    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)

    data_collator = DataCollatorPDTC()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=data_collator, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    print(f"DEBUG: The value of 'learning_rate' right before optimizer setup is: {learning_rate}")

    patient_other_params = [p for n, p in model.patient_encoder.named_parameters() if 'alpha_params' not in n]
    alpha_params_list = [p for n, p in model.patient_encoder.named_parameters() if 'alpha_params' in n]

    assert len(alpha_params_list) > 0, "alpha_params not found in patient_encoder!"

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.drug_model.named_parameters() if p.requires_grad],
            "lr": learning_rate,  
            "weight_decay": weight_decay
        },
        {
            "params": patient_other_params,
            "lr": learning_rate,  
            "weight_decay": weight_decay
        },
        {
            "params": alpha_params_list,
            "lr": learning_rate * 500,
            "weight_decay": 0  
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": learning_rate, 
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.regression_layer.named_parameters()],
            "lr": learning_rate, 
            "weight_decay": weight_decay
        }
    ]

    print("\n--- BEFORE AdaBelief Initialization ---")
    print(f"List's Group 1 LR: {optimizer_grouped_parameters[1]['lr']:.6f}")

    optimizer = AdaBelief(
        optimizer_grouped_parameters,
        # lr=learning_rate,
        eps=1e-16,  
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        weight_decouple=True,  
        rectify=True,  
        print_change_log=False  
    )
    print("\n--- AFTER AdaBelief Initialization ---")
    print(f"Optimizer's Group 1 LR: {optimizer.param_groups[1]['lr']:.6f}")

    for i, group in enumerate(optimizer.param_groups):
        print(f"Optimizer's Group {i} LR: {group['lr']:.6f}")

    print(f"\n--- AFTER Optimizer init, BEFORE Scheduler init ---")
    print(f"Optimizer's Group 1 LR: {optimizer.param_groups[1]['lr']:.6f}")

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    print(f"\n--- AFTER Scheduler init ---")
    print(f"Optimizer's Group 1 LR: {optimizer.param_groups[1]['lr']:.6f}")

    print("--- Parameters managed by optimizer for patient_encoder ---")
    patient_encoder_params = optimizer_grouped_parameters[1]['params']
    patient_encoder_param_names = [n for n, p in model.patient_encoder.named_parameters()]
    for name in patient_encoder_param_names:
        if "alpha_params" in name:
            print(f"FOUND: {name}")

    print(f"Learning rate for this group: {optimizer_grouped_parameters[1]['lr']}")

    if use_mixed_precision and device != 'cpu':
        from torch.amp import GradScaler 
        scaler = GradScaler()  
    else:
        scaler = None

    best_val_loss = float('inf')
    best_val_predictions = None
    best_val_targets = None

    train_targets, train_predictions = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_targets, train_predictions = [], []

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
        all_features = []
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
                        features = model.get_features(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            afm_features=afm_features,
                            adj_features=adj_features,
                            patient_encoding=patient_encoding,
                            PPI_matrix=PPI_matrix,
                            patient_features=patient_features,
                        )

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
                    features = model.get_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                all_features.append(features.cpu().numpy())
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

    if save_importance:
        all_atom_importance = []
        all_atom_pairwise_attentions = []
        all_patient_pairwise_attention = []
        all_patient_sequence_importance = []

        for batch in tqdm(val_loader, desc="Extracting Atom Importance"):
            importance_scores, atom_pairwise_attentions, patient_pairwise_attention, patient_sequence_importance = extract_atom_importance_from_classifier(
                model, batch)

            all_atom_importance.extend(importance_scores)
            all_atom_pairwise_attentions.append(atom_pairwise_attentions.detach().cpu())
            all_patient_pairwise_attention.append(patient_pairwise_attention.detach().cpu())
            all_patient_sequence_importance.append(patient_sequence_importance.detach().cpu())
        pd.DataFrame(all_atom_importance).to_csv(os.path.join(output_dir, f'atom_importance_{run_name}.csv'),
                                                 index=False)
        atom_parwise = torch.cat(all_atom_pairwise_attentions, dim=0)
        pairwise_attention = torch.cat(all_patient_pairwise_attention, dim=0)
        sequence_importance = torch.cat(all_patient_sequence_importance, dim=0)

        print(f"Concatenated pairwise attention tensor shape: {pairwise_attention.shape}")
        print(f"Concatenated sequence importance tensor shape: {sequence_importance.shape}")
        final_atom_pairwise_attention_np = atom_parwise.numpy()
        final_pairwise_attention_np = pairwise_attention.numpy()
        final_sequence_importance_np = sequence_importance.numpy()

        atom_pairwise_attention_path = os.path.join(output_dir, "atom_pairwise_attention.npy")
        sequence_importance_path = os.path.join(output_dir, "sequence_importance.npy")

        print(f"Saving Atompairwise attention to {atom_pairwise_attention_path}...")
        np.save(atom_pairwise_attention_path, final_atom_pairwise_attention_np)
        print(f"Saving sequence importance to {sequence_importance_path}...")
        np.save(sequence_importance_path, final_sequence_importance_np)

        print("Saving complete.")

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
    print(f"The metrics have been recorded {csv_path}")

    if save_features:
        all_features = np.concatenate(all_features, axis=0)

        feature_save_path = f'{run_name}_features_epoch_last.npy'
        label_save_path = f'{run_name}_labels_epoch_last.npy'

        np.save(feature_save_path, all_features)
        np.save(label_save_path, val_targets)
        print(f"Features saved to: {feature_save_path}")
        print(f"Labels saved to: {label_save_path}")

    if draw_tsne:
        print(f"Running t-SNE on {all_features.shape[0]} samples with dimension {all_features.shape[1]}...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
        features_2d = tsne.fit_transform(all_features)
        print("t-SNE finished.")
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=val_targets, 
            cmap='viridis',
            alpha=0.7,
            s=15  
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Ground Truth Label Value', rotation=270, labelpad=15)

        plt.title('t-SNE Visualization of Drug-Patient Combined Features', fontsize=16)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.5)

        image_save_path = os.path.join(output_dir, 'tsne_visualization_epoch_last.png')
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    wandb.finish()
    return model


def get_opt(args, attr_name, default_value):
    if args is None:
        return default_value
    val = getattr(args, attr_name, None)
    return val if val is not None else default_value


def TransferEvaluation(args=None):
    print("Starting the PDTCRegression task.")

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
    project_name = get_opt(args, "project_name", "PDTC")
    num_runs = get_opt(args, "repeat_times", 10) 
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

    for run_idx in range(num_runs):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_run_name = f"run_{run_idx + 1}"
        print(f"Start training: {current_run_name} ({run_idx + 1}/{num_runs})")

        wandb.init(project=project_name, name=current_run_name, reinit=True)

        wandb.config.update({
            "drug_file": drug_file,
            "patient_embed_file": patient_embed_file,
            "sensitivity_file": sensitivity_file,
            "ppi_file": ppi_file,
            "patient_feature_file": patient_embed_file,
            "patient_name_file": patient_name_file,
            "num_classes": num_classes,
            "num_layer": num_layer,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        })

        model = Classifier(
            drug_model_path=model_path,
            patient_encoding_dim=512,
            num_classes=num_classes,
            num_layer=num_layer,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            ParametersNum=893,
        ).to(device)

        trained_model = train_classification_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            weight_decay=weight_decay,
            warmup_steps=int(len(train_dataset) / batch_size * 5),
            project_name=project_name,
            run_name=current_run_name,
            regression_model=regression_model,
            use_mixed_precision=True,
            gradient_accumulation_steps=1,
            num_classes=num_classes,
            csv_path=f"./{project_name}_results/{project_name}_metrics_test_log.csv",
            output_dir=f"./{project_name}_results",
        )
        wandb.finish()

if __name__ == "__main__":
    Regression(args=None)



