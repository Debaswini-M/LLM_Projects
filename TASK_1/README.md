# BERT Fine-Tuning with Custom PyTorch Training Loop & Accelerate

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)
![Accelerate](https://img.shields.io/badge/Accelerate-0.20+-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

This repository contains a complete, low-level implementation of fine-tuning a **BERT (Bidirectional Encoder Representations from Transformers)** model for sequence classification. 

Unlike standard implementations that rely on the high-level `Trainer` API, this project constructs a **raw PyTorch training loop** enhanced with **Hugging Face Accelerate**. This approach demonstrates granular control over the optimization process while maintaining hardware agnosticism (seamless scaling from CPU to GPU/TPU).

## 📄 Project Description

The goal of this project is to detect whether two sentences are paraphrases of each other using the **MRPC (Microsoft Research Paraphrase Corpus)** dataset. The pipeline handles:
1.  Dynamic padding and tokenization.
2.  Distributed-ready data loading.
3.  Custom gradient descent optimization with `AdamW`.
4.  Hardware-aware training using `Accelerator`.

## 📂 Dataset Details

* **Benchmark:** GLUE (General Language Understanding Evaluation)
* **Subset:** MRPC (Microsoft Research Paraphrase Corpus)
* **Input:** Pairs of sentences.
* **Target:** Binary classification (1 = Paraphrase, 0 = Not Paraphrase).

## 🛠️ Tech Stack

This project leverages the following libraries and frameworks to ensure efficient, scalable, and reproducible training:

* **[Python 3.10+](https://www.python.org/)**: The core programming language used for development.
* **[PyTorch](https://pytorch.org/)**: The deep learning backend used for tensor computations and model architecture.
* **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**: Provides the pre-trained `bert-base-uncased` model and tokenization utilities.
* **[Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)**: Abstraction layer to handle device placement (CPU/GPU) and distributed training logic automatically.
* **[Hugging Face Datasets](https://huggingface.co/docs/datasets/index)**: Used for efficient loading and preprocessing of the GLUE (MRPC) benchmark.
* **[Evaluate](https://huggingface.co/docs/evaluate/index)**: A library for easily calculating metrics like Accuracy and F1 Score.
* **[TQDM](https://github.com/tqdm/tqdm)**: Provides progress bars to monitor training epochs and steps.

## 🛠️ Technical Architecture

### Model Configuration
* **Base Model:** `bert-base-uncased` (110M parameters).
* **Head:** `AutoModelForSequenceClassification` with 2 labels.
* **Tokenizer:** `AutoTokenizer` with dynamic truncation.

### Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Batch Size** | 8 |
| **Learning Rate** | 3e-5 |
| **Optimizer** | AdamW |
| **Scheduler** | Linear Decay |
| **Warmup Steps** | 0 |
| **Epochs** | 3 |

## 📊 Performance Results

The model was trained for **3 epochs** on a single GPU setup. The final evaluation on the validation set yielded:

| Metric | Result |
| :--- | :--- |
| **Accuracy** | **86.03%** |
| **F1 Score** | **89.98%** |

*Note: Results may vary slightly due to random seeding and hardware precision.*

## 👩🏻‍💻 Author
**Debaswini-M**
