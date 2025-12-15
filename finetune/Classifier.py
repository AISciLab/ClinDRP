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

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('roberta.'):
                new_key = key[8:]  # len('roberta.') == 8
                new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

        self.drug_model = model

        self.patient_encoder = EvenLayerTransformerEncoder(
            patient_encoding_dim=patient_encoding_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layer,
            num_heads= num_heads,
            ParametersNum=ParametersNum,
        )
        drug_hidden_size = self.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(drug_hidden_size + 768, 512),  
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.regression_layer = nn.Linear(256, 1)  

    def forward(self,
                input_ids=None,
                attention_mask=None,
                afm_features=None,
                adj_features=None,
                patient_encoding=None,
                PPI_matrix=None,
                patient_features=None,
                output_drug_attentions=False):

        drug_outputs = self.drug_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            afm_features=afm_features,
            adj_features=adj_features,
            return_dict=True,
            output_attentions=output_drug_attentions
        )

        drug_representation = drug_outputs.last_hidden_state[:, 0, :]

        patient_output = self.patient_encoder(
            patient_encoding,
            PPI_matrix=PPI_matrix,
            patient_features=patient_features,
            return_attention=output_drug_attentions, 
        )

        if output_drug_attentions:
            patient_representation = patient_output["patient_embedding"]
            patient_pairwise_attention = patient_output["pairwise_attention"]
            patient_sequence_importance = patient_output["sequence_importance"]
        else:
            patient_representation = patient_output  

        combined_representation = torch.cat([drug_representation, patient_representation], dim=1)

        features_256d = self.classifier(combined_representation)

        logits = self.regression_layer(features_256d)  

        if output_drug_attentions:
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

        features_256d = self.classifier(combined_representation)

        return features_256d

