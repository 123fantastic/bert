# BERT 项目说明

这是一个基于 Google Research BERT 实现整理的仓库，主要用于体验和实践 BERT 在自然语言处理任务中的常见用法，包括文本分类、问答任务、特征提取和分词处理。

## 项目简介

BERT（Bidirectional Encoder Representations from Transformers）是一种预训练语言模型，通过大规模文本语料进行预训练，再迁移到下游任务中进行微调。这个仓库保留了经典 BERT 的核心脚本，适合作为学习源码、运行实验和复现基础任务的参考。

## 仓库包含的核心功能

当前仓库中已经包含以下主要脚本：

- `run_classifier.py`：用于文本分类与句对分类任务，例如 MRPC、MNLI、CoLA、XNLI 等。
- `run_squad.py`：用于 SQuAD 1.1 / 2.0 问答任务的训练与预测。
- `extract_features.py`：用于提取 BERT 中间层或最后几层的上下文特征向量。
- `tokenization.py`：实现基础分词、WordPiece 分词以及词表映射。
- `requirements.txt`：项目依赖说明，当前以 TensorFlow 1.11.0 及以上版本为基础。

## 运行环境

建议使用以下环境：

- Python 3
- TensorFlow >= 1.11.0
- 可选 GPU / TPU 环境

安装依赖：

```bash
pip install -r requirements.txt
```

如果你使用 GPU，可以按自己的 CUDA / TensorFlow 兼容版本自行安装对应的 GPU 版本 TensorFlow。

## 预训练模型准备

在运行训练或预测脚本之前，需要先下载对应的 BERT 预训练模型，并准备以下文件：

- `bert_config.json`
- `vocab.txt`
- `bert_model.ckpt`

常见可用模型包括：

- BERT-Base, Uncased
- BERT-Large, Uncased
- BERT-Base, Cased
- BERT-Large, Cased
- BERT-Base, Chinese
- Multilingual BERT

你可以从原始 BERT 发布地址或镜像资源中下载这些模型。

## 文本分类示例

`run_classifier.py` 可用于句子分类或句对分类任务。运行前通常需要准备：

- 数据目录 `data_dir`
- 词表文件 `vocab.txt`
- 配置文件 `bert_config.json`
- 预训练权重 `bert_model.ckpt`
- 输出目录 `output_dir`

示例命令：

```bash
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=./data/mrpc \
  --vocab_file=./model/vocab.txt \
  --bert_config_file=./model/bert_config.json \
  --init_checkpoint=./model/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./output/mrpc
```

## 问答任务示例

`run_squad.py` 用于 SQuAD 数据集上的训练和预测，适合阅读 BERT 在机器阅读理解任务上的处理流程。

训练示例：

```bash
python run_squad.py \
  --vocab_file=./model/vocab.txt \
  --bert_config_file=./model/bert_config.json \
  --init_checkpoint=./model/bert_model.ckpt \
  --do_train=true \
  --train_file=./squad/train-v1.1.json \
  --do_predict=true \
  --predict_file=./squad/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=./output/squad
```

## 特征提取示例

`extract_features.py` 可用于将一句话或句对输入到 BERT 中，导出对应 token 的上下文表示，适合做特征分析或下游模型输入。

示例命令：

```bash
echo "Who was Jim Henson ? ||| Jim Henson was a puppeteer" > input.txt

python extract_features.py \
  --input_file=input.txt \
  --output_file=output.jsonl \
  --vocab_file=./model/vocab.txt \
  --bert_config_file=./model/bert_config.json \
  --init_checkpoint=./model/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

## 分词说明

`tokenization.py` 提供了以下能力：

- 基础文本清洗
- 英文小写化与去重音符号
- 中文字符切分支持
- 标点拆分
- WordPiece 分词
- token 与 id 的双向映射

如果你要自定义数据预处理流程，这个文件通常是首先需要阅读的部分。

## 使用建议

这个仓库更适合以下场景：

- 学习 BERT 原始实现结构
- 了解经典 NLP 任务的微调流程
- 在已有预训练模型基础上做实验
- 研究分词和特征提取细节

需要注意的是，这套代码基于较早版本的 TensorFlow，若你计划在较新的 Python / TensorFlow 环境中长期使用，可能还需要额外做兼容性调整。

## 参考论文

如果你在学习或使用 BERT，建议阅读原始论文：

```bibtex
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

## 许可证

本项目代码与模型遵循 Apache 2.0 License。具体请查看仓库中的许可证文件。

## 说明

本 README 已整理为中文说明，目的是让仓库结构、脚本用途和基本使用方式更容易理解。如果你希望，我下一步还可以继续帮你：

- 补充更完整的中文使用教程
- 增加中文注释版快速开始
- 整理为更适合课程/作业展示的 README 结构
- 顺手补一个更现代的环境说明
