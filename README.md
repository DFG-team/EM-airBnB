# Project 3: Vertex Clustering Algorithm

Vertex is a Wrapper Induction system developed at Yahoo! for extracting structured records from template-based Web pages. Vertex employs a host of novel algorithms for (1) Grouping similar structured pages in a Web site, (2) Picking the appropriate sample pages for wrapper inference, (3) Learning XPath-based extraction rules that are robust to variations in site structure, (4) Detecting site changes by monitoring sample pages, and (5) Optimizing editorial costs by reusing rules

This project contains the implementation of Vertex Clustering algorithm for the first task. In section III, paragraph A, it is described how to group similar structured pages in theory and explains also the pseudocode on which this project is based. 

Requirements
------------

This project requires some specific modules:

 * multiprocessing (https://docs.python.org/3/library/multiprocessing.html)
 * logging (https://docs.python.org/3/library/logging.html)
 * itertools (https://docs.python.org/3/library/itertools.html)
 * beautifulSoup (https://pypi.org/project/beautifulsoup4/)
 * hashlib (https://docs.python.org/3/library/hashlib.html)

How to use
------------

Firstly open a terminal session inside project directory and execute ```pip install -r requirements.txt```, this will install any necessery module not present in current python installation.

Using an IDE, like vscode, open pipeline.py and run. It will ask an user input for which dataset to use. During the execution it will show execution statistics and evaluation metrics

Testing
------------

Using an IDE, like vscode, open pipelineTest.py and run. Using multiprocessing it will start three unique pipeline, one for each dataset.

Maintainers
------------

- Andrea Giorgi
- Filippo De Marco
