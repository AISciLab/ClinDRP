import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.amp import autocast  # 推荐用法
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
    """
    计算多个回归评价指标

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        包含多个指标的元组
    """
    # 将数据转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算MSE和RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # 计算EVS (Explained Variance Score)
    evs = 1 - np.var(y_true - y_pred) / np.var(y_true)

    # 计算MedAE (Median Absolute Error)
    medae = np.median(np.abs(y_true - y_pred))

    # 计算MaxAE (Maximum Absolute Error)
    maxae = np.max(np.abs(y_true - y_pred))

    # 计算MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 计算相关系数
    pearson_corr, _ = stats.pearsonr(y_true, y_pred)
    spearman_corr, _ = stats.spearmanr(y_true, y_pred)

    # 计算MAE和R2
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
                               output_dir="results",  # 指定输出目录
                               save_every_epoch_preds=False,  # 是否保存每一轮的预测
                               save_best_preds=False,
                               csv_path="metrics_log.csv",
                               draw_tsne=False,
                               save_features=False, save_importance=False):
    """
    训练分类模型，支持混合精度训练

    参数:
        model: 组合模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        weight_decay: 权重衰减
        warmup_steps: 预热步数
        project_name: wandb项目名称
        run_name: wandb运行名称
        use_mixed_precision: 是否使用混合精度训练
        gradient_accumulation_steps: 梯度累积步数
        num_classes: 分类类别数
        output_dir: 输出目录
        save_every_epoch_preds: 是否保存每一轮的预测结果
        save_best_preds: 是否保存最佳预测结果
        csv_path: 保存指标的CSV路径
    """
    # # 初始化wandb
    # wandb.init(project=project_name, name=run_name)
    # 记录超参数

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

    # --- 新增: 创建输出目录 ---
    os.makedirs(output_dir, exist_ok=True)

    # 将模型移至设备
    model = model.to(device)

    # 创建数据加载器
    data_collator = DataCollatorPDTC()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=data_collator, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    # 准备优化器和学习率调度器
    # 分层学习率 - 为不同组件设置不同的学习率
    print(f"DEBUG: The value of 'learning_rate' right before optimizer setup is: {learning_rate}")

    patient_other_params = [p for n, p in model.patient_encoder.named_parameters() if 'alpha_params' not in n]
    alpha_params_list = [p for n, p in model.patient_encoder.named_parameters() if 'alpha_params' in n]

    # 确认我们找到了alpha_params
    assert len(alpha_params_list) > 0, "alpha_params not found in patient_encoder!"

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.drug_model.named_parameters() if p.requires_grad],
            "lr": learning_rate,  # 药物模型较小的学习率
            "weight_decay": weight_decay
        },
        {
            "params": patient_other_params,
            "lr": learning_rate,  # 患者编码器较大的学习率
            "weight_decay": weight_decay
        },
        {
            "params": alpha_params_list,
            "lr": learning_rate * 500,
            "weight_decay": 0  # 门控参数通常不使用权重衰减
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": learning_rate,  # 分类器标准学习率
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.regression_layer.named_parameters()],
            "lr": learning_rate,  # 分类器标准学习率
            "weight_decay": weight_decay
        }
    ]

    # 1. 在优化器初始化之前检查
    print("\n--- BEFORE AdaBelief Initialization ---")
    print(f"List's Group 1 LR: {optimizer_grouped_parameters[1]['lr']:.6f}")

    optimizer = AdaBelief(
        optimizer_grouped_parameters,
        # lr=learning_rate,
        eps=1e-16,  # Transformer模型推荐使用1e-16
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        weight_decouple=True,  # 新版本默认值
        rectify=True,  # 新版本默认值
        print_change_log=False  # 关闭警告信息
    )
    # 3. 在优化器初始化之后检查优化器内部的状态
    print("\n--- AFTER AdaBelief Initialization ---")
    # 优化器会将参数组存储在 self.param_groups 属性中
    print(f"Optimizer's Group 1 LR: {optimizer.param_groups[1]['lr']:.6f}")

    # 让我们检查所有组
    for i, group in enumerate(optimizer.param_groups):
        print(f"Optimizer's Group {i} LR: {group['lr']:.6f}")

    print(f"\n--- AFTER Optimizer init, BEFORE Scheduler init ---")
    print(f"Optimizer's Group 1 LR: {optimizer.param_groups[1]['lr']:.6f}")

    # 更新学习率调度器的总步数，考虑梯度累积
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    print(f"\n--- AFTER Scheduler init ---")
    print(f"Optimizer's Group 1 LR: {optimizer.param_groups[1]['lr']:.6f}")

    # 在定义完 optimizer_grouped_parameters 之后
    print("--- Parameters managed by optimizer for patient_encoder ---")
    # 假设 patient_encoder 是第1个参数组 (索引为1)
    patient_encoder_params = optimizer_grouped_parameters[1]['params']
    # 你需要从 model.named_parameters() 中反向查找名称，这有点复杂
    # 一个更简单的方法是直接打印模型中的参数名
    patient_encoder_param_names = [n for n, p in model.patient_encoder.named_parameters()]
    for name in patient_encoder_param_names:
        if "alpha_params" in name:
            print(f"FOUND: {name}")

    # 同时检查学习率
    print(f"Learning rate for this group: {optimizer_grouped_parameters[1]['lr']}")

    # 设置混合精度训练
    if use_mixed_precision and device != 'cpu':
        from torch.amp import GradScaler  # 推荐用法
        scaler = GradScaler()  # 指定设备
    else:
        scaler = None

    # 添加早停机制
    best_val_loss = float('inf')

    best_val_predictions = None
    best_val_targets = None

    train_targets, train_predictions = [], []

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_targets, train_predictions = [], []

        # 训练循环
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        step = 0

        for batch in progress_bar:
            # 将数据移至设备
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

            # 清除梯度
            optimizer.zero_grad()

            # 使用混合精度训练
            if use_mixed_precision and device != 'cpu':
                with autocast("cuda"):
                    # 前向传播
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        afm_features=afm_features,
                        adj_features=adj_features,
                        patient_encoding=patient_encoding,
                        PPI_matrix=PPI_matrix,
                        patient_features=patient_features,
                    )
                    # 计算损失
                    loss = F.mse_loss(outputs.squeeze(-1), labels)  # 使用均方误差损失
                # 使用scaler进行反向传播和参数更新
                scaler.scale(loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # 更新权重
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # 清除梯度
                    scheduler.step()  # 更新学习率
            else:
                # 标准的前向传播和反向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    afm_features=afm_features,
                    adj_features=adj_features,
                    patient_encoding=patient_encoding,
                    PPI_matrix=PPI_matrix,
                    patient_features=patient_features,
                )
                # 计算损失
                loss = F.mse_loss(outputs.squeeze(-1), labels)
                loss.backward()
                # 只有在完成了梯度累积步数后才更新模型参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            step += 1
            train_loss += loss.item() * gradient_accumulation_steps  # 调整回实际损失值
            train_targets.extend(labels.cpu().numpy())
            train_predictions.extend(outputs.squeeze(-1).detach().cpu().numpy())
            progress_bar.set_postfix({'loss': loss.item()})
            # 记录每个batch的损失
            wandb.log({"batch_loss": loss.item()})

        # 计算训练指标
        train_avg_loss = train_loss / len(train_loader)
        train_metrics = calculate_regression_metrics(train_targets, train_predictions)

        # 记录到wandb
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
        # 验证
        model.eval()
        val_loss = 0
        val_targets, val_predictions = [], []
        all_features = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 将数据移至设备
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

                # 前向传播
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
        # 记录到wandb
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
        # 打印结果
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

        # 1. 如果设置了保存每一轮的预测
        if save_every_epoch_preds:
            preds_df = pd.DataFrame({
                'true_label': val_targets,
                'predicted_value': val_predictions
            })
            preds_save_path = os.path.join(output_dir, f'preds_{run_name}_epoch_{epoch + 1}.csv')
            preds_df.to_csv(preds_save_path, index=False)
            print(f"Epoch {epoch + 1} predictions saved to {preds_save_path}")

        # 2. 检查是否是最佳模型，并保存模型和预测结果
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            # 如果设置了保存最佳预测，则更新最佳预测列表
            if save_best_preds:
                best_val_predictions = val_predictions[:]  # 使用切片创建副本
                best_val_targets = val_targets[:]  # 使用切片创建副本
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
        # 2. 准备存储结果
        all_atom_importance = []
        all_atom_pairwise_attentions = []
        all_patient_pairwise_attention = []
        all_patient_sequence_importance = []

        # 3. 遍历数据并提取重要性
        for batch in tqdm(val_loader, desc="Extracting Atom Importance"):
            # 使用新函数提取分数
            importance_scores, atom_pairwise_attentions, patient_pairwise_attention, patient_sequence_importance = extract_atom_importance_from_classifier(
                model, batch)

            # 存储结果
            all_atom_importance.extend(importance_scores)
            all_atom_pairwise_attentions.append(atom_pairwise_attentions.detach().cpu())
            all_patient_pairwise_attention.append(patient_pairwise_attention.detach().cpu())
            all_patient_sequence_importance.append(patient_sequence_importance.detach().cpu())
            print(patient_pairwise_attention.shape)
            print(patient_sequence_importance.shape)

        print(f"成功为 {len(all_atom_importance)} 个分子提取了重要性分数。")
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

        # 使用 np.save 保存数组
        print(f"Saving Atompairwise attention to {atom_pairwise_attention_path}...")
        np.save(atom_pairwise_attention_path, final_atom_pairwise_attention_np)
        print(f"Saving sequence importance to {sequence_importance_path}...")
        np.save(sequence_importance_path, final_sequence_importance_np)

        print("Saving complete.")

    # 定义要保存的字典
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

    # 检查文件是否存在
    file_exists = os.path.isfile(csv_path)

    # 转成 DataFrame 并保存
    df = pd.DataFrame([log_data])

    if file_exists:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"指标已记录到 {csv_path}")

    if save_features:
        all_features = np.concatenate(all_features, axis=0)

        feature_save_path = f'{run_name}_features_epoch_last.npy'
        label_save_path = f'{run_name}_labels_epoch_last.npy'

        np.save(feature_save_path, all_features)
        np.save(label_save_path, val_targets)
        print(f"Features saved to: {feature_save_path}")
        print(f"Labels saved to: {label_save_path}")

    if draw_tsne:
        # 3. 使用 t-SNE 进行降维
        print(f"Running t-SNE on {all_features.shape[0]} samples with dimension {all_features.shape[1]}...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
        features_2d = tsne.fit_transform(all_features)
        print("t-SNE finished.")

        # 4. 绘制散点图
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=val_targets,  # 使用真实标签值作为颜色
            cmap='viridis',  # 'viridis', 'plasma', 'coolwarm' 都是不错的选择
            alpha=0.7,
            s=15  # 点的大小
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
    print("训练完成!")
    return model


def get_opt(args, attr_name, default_value):
    """
    辅助函数：获取参数值
    逻辑：
    1. 如果 args 为 None，返回默认值
    2. 如果 args 中没有该属性，返回默认值
    3. 如果 args 中该属性值为 None，返回默认值
    4. 否则，返回 args 中的值
    """
    if args is None:
        return default_value
    val = getattr(args, attr_name, None)
    return val if val is not None else default_value


def TransferEvaluation(args=None):
    print("开始运行 PDTCRegression 任务")

    # ================= 配置区域 =================
    # 路径配置
    model_path = get_opt(args, "model_path", "./model/multiSmiles_model/checkpoint-80000")
    tokenizer_path = get_opt(args, "tokenizer_path", "./model/custom_tokenizer")

    # 数据路径
    drug_file = get_opt(args, "drug_file", 'data/PDTC/PDTC_Drug_output.csv')
    ppi_file = get_opt(args, "sample_ppi_file", 'data/PDTC/PDTC_PPI.npy')

    patient_embed_file = get_opt(args, "sample_patient_embed_file", 'data/PDTC/PDTCSample_893_embeddings.npy')
    sensitivity_file = get_opt(args, "sample_sensitivity_file", 'data/PDTC/SampleResponse.csv')
    patient_feature_file = get_opt(args, "sample_patient_feature_file", 'data/PDTC/PDTCSample_893_laplacian.npy')
    patient_name_file = get_opt(args, "sample_patient_name_file", 'data/PDTC/PDTCSample_893_normed.csv')

    Model_patient_embed_file = get_opt(args, "model_patient_embed_file", 'data/PDTC/PDTCModel_893_embeddings.npy')
    Model_sensitivity_file = get_opt(args, "model_sensitivity_file", "data/PDTC/ModelResponse.csv")
    Model_patient_feature_file = get_opt(args, "model_patient_feature_file",
                                         'data/PDTC/PDTCModel_893_laplacian.npy')
    Model_patient_name_file = get_opt(args, "model_patient_name_file", 'data/PDTC/PDTCModel_893_normed.csv')

    # 模型与训练超参数
    num_classes = get_opt(args, "num_classes", 1)
    num_layer = get_opt(args, "num_layer", 2)
    num_heads = get_opt(args, "num_heads", 4)
    dropout_rate = get_opt(args, "dropout_rate", 0.0)
    batch_size = get_opt(args, "batch_size", 120)
    num_epochs = get_opt(args, "num_epochs", 150)
    learning_rate = get_opt(args, "learning_rate", 1e-4)
    weight_decay = get_opt(args, "weight_decay", 0.0001)
    project_name = get_opt(args, "project_name", "PDTC")
    # 运行控制
    num_runs = get_opt(args, "repeat_times", 10)  # 默认跑10次
    regression_model = get_opt(args, "regression_model", None)

    # ================= 逻辑开始 =================

    print(f"当前配置 -> Batch: {batch_size}, LR: {learning_rate}, Epochs: {num_epochs}")
    print(f"数据路径 -> {drug_file}")

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
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")

    for run_idx in range(num_runs):
        print(f"训练集大小: {train_size}")
        print(f"验证集大小: {val_size}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_run_name = f"run_{run_idx + 1}"
        print(f"开始训练: {current_run_name} ({run_idx + 1}/{num_runs})")

        # 初始化wandb
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
        print(f"模型 {current_run_name} 微调完成！")
        wandb.finish()

    print("所有训练完成!")


if __name__ == "__main__":
    Regression(args=None)
