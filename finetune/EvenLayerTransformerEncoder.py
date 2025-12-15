import torch
from torch import nn
import torch.nn.functional as F


class EvenLayerTransformerEncoder(nn.Module):
    def __init__(self, patient_encoding_dim, dropout_rate=0.05, num_layers=4, num_heads=8, ParametersNum=933):
        super().__init__()

        assert num_layers % 2 == 0, "Number of layers must be even"

        self.input_projection = nn.Linear(patient_encoding_dim, 512)

        self.alpha_params = nn.ParameterList([
            nn.Parameter(torch.empty(ParametersNum, ParametersNum)) for _ in range(num_layers)
        ])

        for param in self.alpha_params:
            nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')

        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=512,
                nhead=num_heads,
                dim_feedforward=1024,
                dropout=dropout_rate,
                activation=F.relu,
                layer_idx=i
            ) for i in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(512) for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.seq_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, patient_encoding, PPI_matrix=None, patient_features=None, return_attention=False):
        x = self.input_projection(patient_encoding)

        all_attention_weights = []

        for i, (layer, layer_norm, alpha_matrix) in enumerate(
                zip(self.transformer_layers, self.layer_norms, self.alpha_params)):
            external_matrix = None
            if i % 2 == 0:
                if patient_features is not None:
                    external_matrix = patient_features
            elif PPI_matrix is not None:
                external_matrix = PPI_matrix

            layer_output = layer(x, external_matrix=external_matrix, alpha_matrix=alpha_matrix,
                                 return_attention=return_attention)
            if return_attention:
                x, attention_weights = layer_output
                all_attention_weights.append(attention_weights)
            else:
                x = layer_output

            x = layer_norm(x)  # [batch_size, 1200, 512]


        pooled_output = x.transpose(1, 2)  # [batch_size, 512, 1200]
        pooled_output = self.seq_pooling(pooled_output)  # [batch_size, 512, 1]
        pooled_output = pooled_output.squeeze(2)  # [batch_size, 512]
        final_embedding = self.output_projection(pooled_output)

        if not return_attention:
            return final_embedding

        else:
            stacked_attentions = torch.stack(all_attention_weights, dim=0)

            # [num_layers, batch, nhead, seq, seq] -> [batch, seq, seq]
            final_pairwise_attention = stacked_attentions.mean(dim=[0, 2])

            sequence_importance = final_pairwise_attention.sum(dim=1)  # [batch, seq_len]

            return {
                "patient_embedding": final_embedding,  # [batch_size, 768]
                "pairwise_attention": final_pairwise_attention,  # [batch_size, seq_len, seq_len]
                "sequence_importance": sequence_importance  # [batch_size, seq_len]
            }


class CustomTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_rate = dropout

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, src, external_matrix=None, alpha_matrix=None, return_attention=False):
        batch_size, seq_len, _ = src.shape

        if external_matrix is not None or alpha_matrix is not None:
            attn_output, attn_weights = self._optimized_attention(src, external_matrix, alpha_matrix)
        else:
            attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)

        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.activation(self.linear1(src))
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)

        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        if return_attention:
            return src, attn_weights
        else:
            return src

    def _optimized_attention(self, src, external_matrix, alpha_matrix):
        batch_size, seq_len, _ = src.shape

        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)

        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q / self.head_dim ** 0.5
        dynamic_attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, nhead, seq_len, seq_len]

        if external_matrix is not None:
            normalized_ext_matrix = external_matrix
            # [batch_size, seq_len, seq_len]

            alpha_gate = torch.sigmoid(alpha_matrix)
            alpha_gate_expanded = alpha_gate.unsqueeze(0).unsqueeze(0)

            ext_matrix_expanded = normalized_ext_matrix.unsqueeze(1)

            raw_attn_scores = (dynamic_attn_scores * alpha_gate_expanded +
                               (1 - alpha_gate_expanded) * ext_matrix_expanded)
        else:
            raw_attn_scores = dynamic_attn_scores

        soft_attn_weights = F.softmax(raw_attn_scores, dim=-1)

        if self.dropout_rate > 0:
            soft_attn_weights = F.dropout(soft_attn_weights, p=self.dropout_rate)

        attn_output = torch.matmul(soft_attn_weights, v)  # [batch_size, nhead, seq_len, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        attn_output = self.out_proj(attn_output)


        return attn_output, raw_attn_scores
