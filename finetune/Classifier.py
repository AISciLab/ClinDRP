import torch
from torch import nn

from finetune.EvenLayerTransformerEncoder import EvenLayerTransformerEncoder
from PretrainModel.MultiSmilesModel import MultiSmilesModelConfig, MultiSmilesModel


class Classifier(nn.Module):
    def __init__(self, drug_model_path, patient_encoding_dim, num_classes=4, num_layer=4,
                 num_heads = 8,dropout_rate=0.1,ParametersNum=933,):
        super().__init__()

        self.config = MultiSmilesModelConfig.from_pretrained(drug_model_path)# 加载预训练的多模态药物模型配置

        model = MultiSmilesModel(self.config)

        state_dict = torch.load(f'{drug_model_path}/pytorch_model.bin')

        # 创建新的状态字典，移除一个roberta前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('roberta.'):
                new_key = key[8:]  # len('roberta.') == 8
                new_state_dict[new_key] = value

        # 加载处理后的权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        self.drug_model = model

        # 使用偶数层 Transformer 患者编码器
        self.patient_encoder = EvenLayerTransformerEncoder(
            patient_encoding_dim=patient_encoding_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layer,
            num_heads= num_heads,
            ParametersNum=ParametersNum,
        )

        # 获取药物表示的维度
        drug_hidden_size = self.config.hidden_size

        # 结合层和分类层
        self.classifier = nn.Sequential(
            nn.Linear(drug_hidden_size + 768, 512),  # 结合药物和患者表示
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.regression_layer = nn.Linear(256, 1)  # 用于回归任务的输出层

    def forward(self,
                input_ids=None,
                attention_mask=None,
                afm_features=None,
                adj_features=None,
                patient_encoding=None,
                PPI_matrix=None,
                patient_features=None,
                output_drug_attentions=False):
        # 获取药物表示
        drug_outputs = self.drug_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            afm_features=afm_features,
            adj_features=adj_features,
            return_dict=True,
            output_attentions=output_drug_attentions
        )

        # 使用[CLS]令牌的表示作为药物的整体表示
        drug_representation = drug_outputs.last_hidden_state[:, 0, :]

        # 处理患者编码，现在PPI_matrix和patient_features都是1200×1200的矩阵
        patient_output = self.patient_encoder(
            patient_encoding,
            PPI_matrix=PPI_matrix,
            patient_features=patient_features,
            return_attention=output_drug_attentions,  # 如果需要注意力输出
        )

        if output_drug_attentions:
            patient_representation = patient_output["patient_embedding"]
            patient_pairwise_attention = patient_output["pairwise_attention"]
            patient_sequence_importance = patient_output["sequence_importance"]
        else:
            patient_representation = patient_output  # 如果不需要注意力输出，直接使用患者表示

        # 结合药物和患者表示
        combined_representation = torch.cat([drug_representation, patient_representation], dim=1)

        # 分类
        features_256d = self.classifier(combined_representation)

        logits = self.regression_layer(features_256d)  # 回归任务的输出

        # 根据是否需要，返回不同的输出
        if output_drug_attentions:
            # 如果需要注意力，返回一个包含所有信息的字典或元组
            from collections import namedtuple
            ModelOutput = namedtuple("ModelOutput", ["logits", "drug_attentions", "drug_text_length","patient_pairwise_attention","patient_sequence_importance"])
            return ModelOutput(
                logits=logits,
                drug_attentions=drug_outputs.attentions,
                drug_text_length=drug_outputs.text_length,
                patient_pairwise_attention=patient_pairwise_attention,
                patient_sequence_importance=patient_sequence_importance
            )
        else:
            # 默认行为，只返回logits，保持与训练代码的兼容性
            return logits

        return logits

    def get_features(self,
                     input_ids=None,
                     attention_mask=None,
                     afm_features=None,
                     adj_features=None,
                     patient_encoding=None,
                     PPI_matrix=None,
                     patient_features=None):

        # 这部分逻辑和 forward 方法里的是完全一样的
        drug_outputs = self.drug_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            afm_features=afm_features,
            adj_features=adj_features,
            return_dict=True
        )
        drug_representation = drug_outputs.last_hidden_state[:, 0, :]
        patient_representation = self.patient_encoder(
            patient_encoding,
            PPI_matrix=PPI_matrix,
            patient_features=patient_features
        )
        combined_representation = torch.cat([drug_representation, patient_representation], dim=1)

        # 计算并返回256维特征
        features_256d = self.classifier(combined_representation)

        return features_256d
