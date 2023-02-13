# A benchmark for toxic comment classification on Civil Comments dataset

## A) Abstract
> Toxic comment detection on social media has proven to be essential for content moderation. This paper compares a wide set of different models on a highly skewed multi-label hate speech dataset. We consider inference time and several metrics to measure performance and bias in our comparison. We show that all BERTs have similar performance regardless of the size, optimizations or language used to pre-train the models. RNNs are much faster at inference than any of the BERT. BiLSTM remains a good compromise between performance and inference time. RoBERTa with Focal Loss offers the best performance on biases and AUROC. However, DistilBERT combines both good AUROC and a low inference time. All models are affected by the bias of associating identities. BERT, RNN, and XLNet are less sensitive than the CNN and Compact Convolutional Transformers.

## B) How to install the virtual environment
### B.1) Recommended method
The recommended method is to use `pyenv` and `poetry`.

To do this, you must already have installed on your machine :
* [Pyenv](https://github.com/pyenv/pyenv)
* [Poetry](https://python-poetry.org/)

Then, you just have to :
* Git clone the project
* Run `pyenv install 3.8.9` to install python 3.8.9 (if not already installed)
* Run `pyenv shell 3.8.9` to use python 3.8.9
* Run `poetry install` in the project folder
* Run `poetry shell` to enable the Python virtual environment.

### B.2) Alternative method
In case you don't have `pyenv` and `poetry`, you must :
* have **Python 3.8.9** installed on your machine
* `virtualenv` on your Python 3.8.9 (for example via `pip`)

Then, you just have to :
* Git clone of the project
* Run `python -m venv .venv` in the project folder. Be careful to choose the right version of python, for example `python3.9 -m venv .venv`
* Run `source ./.venv/bin/activate` to activate the virtual environment
* Run `pip install -r requirements.txt`

## C) Team

| Name             | Email (@epita.fr)         | Github account |
| ---------------- | ------------------------- | -------------- |
| Corentin Duchêne | corentin.duchene          | [`Nigiva`](https://github.com/Nigiva) |
| Henri Jamet      | henri.jamet               | [`hjamet`](https://github.com/hjamet) |
| Pierre Guillaume | pierre.guillaume          | [`drguigui1`](https://github.com/drguigui1) |
| Réda Dehak       | reda.dehak                |                |

## D) Citation
In EGC 2023, vol. RNTI-E-39, pp.19-30.

```
@article{RNTI/papers/1002807,
  author    = {Corentin Duchêne and Henri Jamet and Pierre Guillaume and Réda Dehak},
  title     = {Benchmark pour la classification de commentaires toxiques sur le jeu de données Civil Comments},
  journal = {Revue des Nouvelles Technologies de l'Information},
  volume = {Extraction et Gestion des Connaissances, RNTI-E-39},
  year      = {2023},
  pages     = {19-30}
}
```

## E) License

[**MIT license**](opensource.org/licenses/mit-license.php)
