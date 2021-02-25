# Project 4: Entity Matching applied to AirBnB listings

This project is focused on implementing state-of-art solutions for Entity Matching problem. We define as entity a real object and we define as entity mention a reference to a real word entity. The goal of Entity Matching EM is to find all possibile pairs of entity mentions between two sets, D and D', which refers to the same real object. 
The solutions studied and implemented are:
- DeepMatcher (http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)
- DeepER (https://arxiv.org/pdf/1710.00597.pdf)


Requirements
------------

This project requires some specific modules:

 * gensim (https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py)
 * keras (https://keras.io/api/)
 * numPy (https://numpy.org/doc/)

How to use
------------

Using an IDE, like vscode or PyCharm, open main.py and run. It will esecute the pipeline related to DeepER. In order to execute the pipeline associated to DeepMatcher just comment ```start_pipeline_with_DeepER("DatasetDeepER\\")``` and uncomment ```start_pipeline_summary("DatasetDeepMatcher\\")```.

DeepMatcher implementation is located inside ```DeepMatcher_Train``` folder, there are multiple .ipynb files which can be executed inside Google CoLab.

Maintainers
------------

- Andrea Giorgi
- Pier Vincenzo De Lellis
- Francesco Foresi
