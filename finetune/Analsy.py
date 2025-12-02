import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.amp import autocast  # 推荐用法
from torch.utils.data import DataLoader
from tqdm import tqdm


# 这个新函数将用于您的可解释性分析脚本中
def extract_atom_importance_from_classifier(model, batch):
    """
    从完整的 Classifier 模型中提取原子重要性分数。

    参数:
        model (Classifier): 您训练好的完整模型。
        batch (dict): 包含所有必需输入的批次数据字典。

    返回:
        list[torch.Tensor]: 包含每个分子真实原子重要性分数的列表。
    """
    model.eval()
    with torch.no_grad():
        # 将所有数据移动到设备
        device = next(model.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        # --- 关键步骤：调用 forward 并请求注意力权重 ---
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            afm_features=batch['afm_features'],
            adj_features=batch['adj_features'],
            patient_encoding=batch['patient_encoding'],
            PPI_matrix=batch['ppi_matrix'],
            patient_features=batch['patient_features'],
            output_drug_attentions=True  # <--- 开启注意力输出
        )

        # --- 后续处理逻辑与之前完全相同 ---
        # `outputs.drug_attentions` 就是我们需要的注意力元组
        last_layer_attentions = outputs.drug_attentions[-1]

        cls_token_attentions = last_layer_attentions[:, :, 0, :]

        avg_cls_attentions = cls_token_attentions.mean(dim=1)

        text_length = outputs.drug_text_length
        atom_attentions_unmasked = avg_cls_attentions

        knowledge_mask = torch.any(batch['afm_features'] != 0, dim=-1).int()

        patient_pairwise_attention = outputs.patient_pairwise_attention

        patient_sequence_importance = outputs.patient_sequence_importance

        batch_importance_scores = []
        for i in range(atom_attentions_unmasked.shape[0]):
            sample_scores = atom_attentions_unmasked[i]
            real_atom_scores = sample_scores
            batch_importance_scores.append(real_atom_scores)

        # 这里可以添加对 patient_pairwise_attention 和 patient_sequence_importance 的处理
        # 如果需要，可以将 patient_pairwise_attention 和 patient_sequence_importance 添加到返回的字典中
        # 例如：

    return batch_importance_scores , patient_pairwise_attention, patient_sequence_importance