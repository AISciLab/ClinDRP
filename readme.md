Here is the professional English translation of your README. I have refined the wording to follow standard open-source and academic documentation conventions.

***

# 3DMolReg: Multimodal Drug Response Prediction

This project proposes a deep learning model based on multimodal SMILES sequences and patient characteristics to predict drug response.

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
│   ├── multiSmiles_model/
│   │   └── checkpoint-80000/
│   └── custom_tokenizer/
├── data/                  # Data directory
│   └── PDTC/              # PDTC related data (.csv, .npy)
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
    python main.py pdx-fine
    ```

*   **Custom Execution:**
    ```bash
    # Specify WandB project name and increase number of Epochs
    python main.py pdx-fine \
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

