# Ontology Matching with Machine Learning
Applying of Machine Learning Techniques to Combine String-based, Language-based and Structure-based Similarity Measures for Ontology Matching [[Paper]](http://ceur-ws.org/Vol-2523/paper14.pdf)

## Getting Started

1. Download [Word2Vec model](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) and unzip to root folder.
2. Install requirements.
3. Select dataset and machine learning algorithm in config.yml.
4. Create dataset:
```
python3 create_dataset.py
```
5. Calculate features:
```
python3 calculate_features.py
```
6. Train and evaluate model:
```
python3 train_evaluate.py
```

## Main requirements

* [ngram](https://github.com/gpoulter/python-ngram)
* [fuzzycomp](https://github.com/fuzzycode/fuzzycomp)
* [py_stringmatching](https://sites.google.com/site/anhaidgroup/projects/magellan/py_stringmatching)
* [gensim](https://pypi.org/project/gensim/)
* [nltk](https://www.nltk.org)
* [owlready2](https://pypi.org/project/Owlready2/)

## Instal fuzzycomp for python3

1. Clone repo
```
git clone https://github.com/fuzzycode/fuzzycomp
```
2. From fuzzycomp/fuzzycomp/fuzzycomp.py delete this line
```
from exceptions import IndexError, ValueError
```
3. Install the package
```
python3 setup.py install
```

