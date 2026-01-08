***

# ClinDRP: Multimodal Drug Response Prediction

Accurate identification of drug response in cancer patients is the foundation of precision medicine. Recently, molecular language
models provide the great opportunities for drug response prediction. However, most methods ignore the mapping relation
cross modalities, leading to deviate from the basic criterion of molecular modeling. Furthermore, the clinical response of
patients is closely associated with multi-scale regulatory networks within life systems. In this study, we propose a deep learning
framework, ClinDRP, that integrates cross-modal interaction of molecules and multi-order regulatory within patient to improve
clinical drug response prediction. Specifically, ClinDRP designs a conditional masked language model on the unified multimodal
sequences where three-dimensional (3D) conformations are discretized into tokens as context of 1D sequence, thus capturing
fine-grained interaction cross modalities. In patient representation learning, regulatory networks at gene, protein and pathway
levels are hierarchically embedded into Transformer through adaptive mapping functions. Across multiple scenarios of drug
response and disease progression prediction, ClinDRP achieved superior performance and explored potential mechanisms
behind pharmacological and biochemical processes. Importantly, the sensitive drugs predicted by ClinDRP can improve the
survival outcomes of clinical patients, further highlighting its potential in anticancer drug therapies.

## 🛠 Installation

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

To ensure the code runs seamlessly with default parameters, it is recommended to maintain the following directory structure:

```text
Project/
├── main.py                # Entry point containing launch logic for all tasks
├── model/                 # Stores pretrained model weights and tokenizer
│   ├── model_path/
│   │   └── checkpoint-80000/
│   └── tokenizer/
├── data/                  # Data directory
│   ├── PDTC/              # PDTC related data (.csv, .npy)
│   └── PDX/               # PDX related data (.csv, .npy)
└── README.md
```

---

## 🚀 Usage

This project uses a unified `main.py` entry point and utilizes **Subcommands** to distinguish between different experimental tasks.

The code is designed to support two execution modes:
1.  **Reproduction Mode:** Run without arguments to automatically load the preset default paths and hyperparameters defined in the code.
2.  **Custom Experiment Mode:** Override default configurations via command-line arguments.

### 1. Pretraining
Use this task if you wish to re-pretrain the model.

*   **Default Execution:**
    ```bash
    python main.py Pretrain
    ```

*   **Specify Pretraining Data:**
    ```bash
    python main.py Pretrain \
        --train_data 'data/Pretrain/out_embedding_train.csv'
    ```

### 2. TransferEvaluation Task
Runs the baseline PDTC dataset training task. By default, it trains using `Sample` data and predicts outcomes on `Model` data.

*   **Default Execution:**
    ```bash
    python main.py pdtc-reg
    ```

*   **Custom Execution:**
    ```bash
    # Set batch size to 64, learning rate to 5e-5, and repeat 3 times
    python main.py pdtc-reg \
        --batch_size 64 \
        --learning_rate 5e-5 \
        --repeat_times 3
    ```

### 3. PDTC CrossValidation
Runs a rigorous 10-fold cross-validation experiment to evaluate the model's generalization capability.

*   **Default Execution:**
    ```bash
    python main.py pdtc-ten
    ```

*   **Custom Execution:**
    ```bash
    # Specify WandB project name and increase number of Epochs
    python main.py pdtc-ten \
        --project_name "PDTC_10Fold_Exp1" \
        --num_epochs 200 \
        --batch_size 128
    ```
    
### 4. PDX CrossValidation
Runs a rigorous 10-fold cross-validation experiment to evaluate the model's generalization capability.

*   **Default Execution:**
    ```bash
    python main.py pdx-clas
    ```

*   **Custom Execution:**
    ```bash
    # Specify WandB project name and increase number of Epochs
    python main.py pdx-clas \
        --project_name "PDX_10Fold_Exp1" \
        --num_epochs 200 \
        --batch_size 128
    ```

---

## ⚙️ Arguments

You can view the full list of supported parameters for any task by running `python main.py <task> --help`. Below are the common arguments:

| Argument | Type | Description | Default Value                       |
| :--- | :--- | :--- |:------------------------------------|
| `--model_path` | str | Path to the pretrained model | `./model/multiSmiles_model/...`     |
| `--tokenizer_path` | str | Path to the tokenizer | `./model/custom_tokenizer`          |
| `--batch_size` | int | Training batch size | 128                                 |
| `--learning_rate` | float | Initial learning rate | 1e-4 or 3e-5                        |
| `--num_epochs` | int | Number of training epochs | 150                                 |
| `--dropout_rate` | float | Dropout probability |                                     |
| `--project_name` | str | WandB project name | (Named automatically based on task) |
| `--weight_decay` | float | Weight decay coefficient | 0.0001 / 0.001                      |

---

## 📄 Citation

If you use this project in your research, please cite the following reference:

```bibtex

```






