# Introduction

* **BERT**: (Bidirectional Encoder Representations from Transformers) is a pre-trained natural language processing (NLP) model developed by Google. It's designed to understand context and meaning in text by considering the entire context of a sentence, unlike earlier models which looked at text in a unidirectional manner. 
* **GPT-2**: (Generative Pretrained Transformer) is a powerful language generation model that can generate coherent and contextually relevant text.
* **T5** (Text-to-Text Transfer Transformer) is a model that treats all NLP tasks as text-to-text tasks. It can be applied to a wide range of tasks by framing them as a text-to-text problem.
* **XLNet**: is a transformer model that extends the idea of autoregressive models like GPT but also includes elements of autoencoding. It's designed to capture bidirectional contexts for better performance
* **BART** (BART is a denoising autoencoder for pretraining sequence-to-sequence models) is designed for sequence-to-sequence tasks. It frames tasks as a denoising autoencoder problem, where the model is trained to reconstruct original sequences from noisy versions.
* **ELMo** (Embeddings from Language Models) is a contextual word representation model developed by researchers at the Allen Institute for Artificial Intelligence. It's called "contextual" because unlike traditional word embeddings like Word2Vec or GloVe, which assign a fixed vector to each word in a sentence regardless of its context, ELMo generates a unique vector for each occurrence of a word depending on the context in which it appears.
* **ULMFiT** (Universal Language Model Fine-tuning) is a technique developed by Jeremy Howard and Sebastian Ruder for transfer learning in natural language processing (NLP). It allows for efficient training of high-performance models even when you have a limited amount of task-specific data

# Development setup
## Prerequisites
This following packages must be installed
* python
* poetry
* git

## Configuration
* `poetry` configuration, add environment variable `POETRY_VIRTUALENVS_IN_PROJECT=true`
* `vscode` configuration, add environment variable `PYTHON_VENV_LOC`
  * on windows: `PYTHON_VENV_LOC=.venv\\bin\\python.exe`
  * on linux: `PYTHON_VENV_LOC=.venv/bin/python`
* `git` configuration
```shell
git config --global user.name 'your name'
git config --global user.email 'your email'
```

## Initialization
* First setup `poetry install`
* Then `poetry shell`


## Installation with pip
```shell
pip install --index-url https://test.pypi.org/simple/ mypkg128
```
or
```shell
pip3 install --index-url https://test.pypi.org/simple/ mypkg128
```

# Code of Conduct

# History (changelog)
