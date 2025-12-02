import argparse
import sys

# 假设这些模块在您的项目中存在
from PDXbaseline.PDXFinetune import PDXFinetune
from finetune.TransferEvaluation import TransferEvaluation
from finetune.CrossValidation import CrossValidation
from TCGA.TCGAPredict import TCGA
from PretrainModel.MultiSmilesDataset import Pretrain


def add_common_hyperparameters(parser):
    """注册所有任务通用的超参数"""
    group = parser.add_argument_group('通用超参数')
    group.add_argument("--model_path", type=str, help="预训练模型路径")
    group.add_argument("--tokenizer_path", type=str, help="分词器路径")
    group.add_argument("--num_classes", type=int, help="分类数量")
    group.add_argument("--num_layer", type=int, help="模型层数")
    group.add_argument("--num_heads", type=int, help="注意力头数")
    group.add_argument("--dropout_rate", type=float, help="Dropout率")
    group.add_argument("--batch_size", type=int, help="批次大小")
    group.add_argument("--num_epochs", type=int, help="训练轮数")
    group.add_argument("--learning_rate", type=float, help="学习率")
    group.add_argument("--weight_decay", type=float, help="权重衰减")
    group.add_argument("--project_name", type=str, help="WandB 项目名称")


def add_pdtc_data_args(parser):
    """注册 PDTC 相关的通用数据路径"""
    group = parser.add_argument_group('PDTC 数据路径')
    group.add_argument("--drug_file", type=str, help="药物数据文件")
    group.add_argument("--sample_ppi_file", type=str, help="PPI 网络文件")
    # 样本数据 (Sample)
    group.add_argument("--sample_patient_embed_file", type=str, help="样本患者 Embeddings")
    group.add_argument("--sample_sensitivity_file", type=str, help="样本敏感性文件")
    group.add_argument("--sample_patient_feature_file", type=str, help="样本患者特征(Laplacian)")
    group.add_argument("--sample_patient_name_file", type=str, help="样本患者名称文件")
    # 模型数据 (Model / Validation)
    group.add_argument("--model_patient_embed_file", type=str, help="模型/验证集 Embeddings")
    group.add_argument("--model_sensitivity_file", type=str, help="模型/验证集 敏感性文件")
    group.add_argument("--model_patient_feature_file", type=str, help="模型/验证集 特征")
    group.add_argument("--model_patient_name_file", type=str, help="模型/验证集 名称文件")


def main():
    # 创建顶级解析器
    parser = argparse.ArgumentParser(description="MultiSmilesModel 统一运行入口")
    subparsers = parser.add_subparsers(dest="task", required=True, help="请选择要执行的任务")

    # ===============================================================
    # 任务 1: PDTCRegression (普通回归/重复训练)
    # ===============================================================
    p_reg = subparsers.add_parser("pdtc-reg", help="运行 PDTC 基础回归任务 (支持多次重复)")
    add_common_hyperparameters(p_reg)
    add_pdtc_data_args(p_reg)
    p_reg.add_argument("--repeat_times", type=int, help="重复训练次数 (默认: 10)")

    # ===============================================================
    # 任务 2: PDTCRegressionTenfold (10折交叉验证)
    # ===============================================================
    p_ten = subparsers.add_parser("pdtc-ten", help="运行 PDTC 10折交叉验证")
    add_common_hyperparameters(p_ten)
    add_pdtc_data_args(p_ten)

    # ===============================================================
    # 任务 3: PDXFinetune (PDX 微调)
    # ===============================================================
    p_pdx = subparsers.add_parser("pdx-fine", help="运行 PDX 微调任务")
    add_common_hyperparameters(p_pdx)

    g_pdx = p_pdx.add_argument_group('PDX 数据路径')
    g_pdx.add_argument("--drug_file", type=str, help="PDX 药物文件")
    g_pdx.add_argument("--sample_ppi_file", type=str, help="PDX PPI 文件")
    g_pdx.add_argument("--sample_patient_embed_file", type=str, help="PDX 患者 Embeddings")
    g_pdx.add_argument("--sample_sensitivity_file", type=str, help="PDX 敏感性文件")
    g_pdx.add_argument("--sample_patient_feature_file", type=str, help="PDX 患者特征")
    g_pdx.add_argument("--sample_patient_name_file", type=str, help="PDX 患者名称")
    g_pdx.add_argument("--model_file", type=str, help="PDX 模型文件 (.csv)")

    # ===============================================================
    # 任务 4: TCGA (预测任务)
    # ===============================================================
    p_tcga = subparsers.add_parser("tcga", help="运行 TCGA 预测任务")
    add_common_hyperparameters(p_tcga)
    add_pdtc_data_args(p_tcga)

    g_tcga = p_tcga.add_argument_group('TCGA 数据路径')
    g_tcga.add_argument("--tcga_drug_file", type=str)
    g_tcga.add_argument("--tcga_response_file", type=str)
    g_tcga.add_argument("--tcga_patient_embed_file", type=str)
    g_tcga.add_argument("--tcga_patient_feature_file", type=str)
    g_tcga.add_argument("--tcga_patient_name_file", type=str)
    g_tcga.add_argument("--tcga_output_file", type=str, help="预测结果输出路径")

    # ===============================================================
    # 任务 5: 预训练任务 (Pretrain)
    # ===============================================================
    p_pre = subparsers.add_parser("pretrain", help="运行预训练任务")
    add_common_hyperparameters(p_pre)

    # 预训练特有参数
    g_pre = p_pre.add_argument_group('Pretrain 数据路径')
    g_pre.add_argument("--train_data", type=str,
                       default='data/Pretrain/out_embedding_train.csv',
                       help="训练数据路径")
    g_pre.add_argument("--eval_data", type=str,
                       default='.data/Pretrain/out_embedding_eval.csv',
                       help="验证数据路径")
    g_pre.add_argument("--Pretrain_output_dir", type=str,
                       default="model/Mulit_roberta_model",
                       help="预训练模型输出目录")

    # ===============================================================
    # 解析参数并分发
    # ===============================================================
    args = parser.parse_args()

    # 根据 task 名称调用不同的函数
    if args.task == "pdtc-reg":
        TransferEvaluation(args)
    elif args.task == "pdtc-ten":
        CrossValidation(args)
    elif args.task == "pdx-fine":
        PDXFinetune(args)
    elif args.task == "tcga":
        TCGA(args)
    elif args.task == "pretrain":
        Pretrain(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()