import argparse
import sys

from PDXbaseline.PDXFinetune import PDXFinetune
from finetune.TransferEvaluation import TransferEvaluation
from finetune.CrossValidation import CrossValidation
from PretrainModel.MultiSmilesDataset import Pretrain


def add_common_hyperparameters(parser):
    group = parser.add_argument_group('hyperparameters')
    group.add_argument("--model_path", type=str, help="Pre-trained model path")
    group.add_argument("--tokenizer_path", type=str, help="Tokenizer path")
    group.add_argument("--num_classes", type=int, help="Number of categories")
    group.add_argument("--num_layer", type=int, help="Number of model layers")
    group.add_argument("--num_heads", type=int, help="Number of attention heads")
    group.add_argument("--dropout_rate", type=float, help="Dropout rate")
    group.add_argument("--batch_size", type=int, help="Batch size")
    group.add_argument("--num_epochs", type=int, help="Number of training epochs")
    group.add_argument("--learning_rate", type=float, help="Learning rate")
    group.add_argument("--weight_decay", type=float, help="Weight decay")
    group.add_argument("--project_name", type=str, help="WandB Project Name")


def add_pdtc_data_args(parser):
    group = parser.add_argument_group('PDTC Data path')
    group.add_argument("--drug_file", type=str, help="Drug data file")
    group.add_argument("--sample_ppi_file", type=str, help="PPI Network files")
    group.add_argument("--sample_patient_embed_file", type=str, help="Sample patients Embeddings")
    group.add_argument("--sample_sensitivity_file", type=str, help="Sample Sensitivity file")
    group.add_argument("--sample_patient_feature_file", type=str, help="Sample patient feature(Laplacian)")
    group.add_argument("--sample_patient_name_file", type=str, help="Sample Patient file")
    group.add_argument("--model_patient_embed_file", type=str, help="Moedl patients Embeddings")
    group.add_argument("--model_sensitivity_file", type=str, help="Moedl Sensitivity file")
    group.add_argument("--model_patient_feature_file", type=str, help="Model patient feature")
    group.add_argument("--model_patient_name_file", type=str, help="Model Patient file"")


def main():
    parser = argparse.ArgumentParser(description="MultiSmilesModel Entrance")
    subparsers = parser.add_subparsers(dest="task", required=True, help="Select the task")

    # ===============================================================
    # Task 1: PDTCRegression 
    # ===============================================================
    p_reg = subparsers.add_parser("pdtc-reg", help="Run the PDTC basic regression task")
    add_common_hyperparameters(p_reg)
    add_pdtc_data_args(p_reg)
    p_reg.add_argument("--repeat_times", type=int, help="Number of training iterations (default: 10)")

    # ===============================================================
    # Task 2: PDTCRegressionTenfold
    # ===============================================================
    p_ten = subparsers.add_parser("pdtc-ten", help="Run PDTC 10-fold cross-validation.")
    add_common_hyperparameters(p_ten)
    add_pdtc_data_args(p_ten)

    # ===============================================================
    # 任务 3: PDXFinetune
    # ===============================================================
    p_pdx = subparsers.add_parser("pdx-fine", help="Run the PDX fine-tuning task.")
    add_common_hyperparameters(p_pdx)

    g_pdx = p_pdx.add_argument_group('PDX data path')
    g_pdx.add_argument("--drug_file", type=str, help="PDX Drug Documents")
    g_pdx.add_argument("--sample_ppi_file", type=str, help="PDX PPI file")
    g_pdx.add_argument("--sample_patient_embed_file", type=str, help="PDX patient Embeddings")
    g_pdx.add_argument("--sample_sensitivity_file", type=str, help="PDX Sensitive file")
    g_pdx.add_argument("--sample_patient_feature_file", type=str, help="PDX Patient feature")
    g_pdx.add_argument("--sample_patient_name_file", type=str, help="PDX Patient bame")
    g_pdx.add_argument("--model_file", type=str, help="PDX model file (.csv)")

    # ===============================================================
    # Task 5: Pretrain
    # ===============================================================
    p_pre = subparsers.add_parser("pretrain", help="Pretrain")
    add_common_hyperparameters(p_pre)

    g_pre = p_pre.add_argument_group('Pretrain Data path')
    g_pre.add_argument("--train_data", type=str,
                       default='data/Pretrain/out_embedding_train.csv',
                       help="Training data path")
    g_pre.add_argument("--eval_data", type=str,
                       default='.data/Pretrain/out_embedding_eval.csv',
                       help="Verify data path")
    g_pre.add_argument("--Pretrain_output_dir", type=str,
                       default="model/Mulit_roberta_model",
                       help="Pre-trained model output directory")

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

