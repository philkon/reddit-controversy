## Analysis and Prediction of Multilingual Controversy on Reddit
This repository contains the code used in the paper *Analysis and Prediction of Multilingual Controversy on Reddit*.

### Requirements
Experiments were conducted with Python 3.8.2 and additional Python packages can be found in and installed with the *requirements.txt* file (`pip3 install requirements.txt`). As we are dealing with over 120 million comments, make sure your computer has enough RAM (at least 256 GB recommended).

### Data
You need to download submission and comment files from January 2019 to Dezember 2019 from [PushShift Servers](https://files.pushshift.io/reddit) and put *\*.zst* files in the accoring directories (submissions go to *data/raw/submissions/* and comments go to *data/raw/comments/*).

### Run the code
Run each *.py* file according to the number in the beginning of the file name.

* **0\_setup.py:** Creates all the directories needed for execution.
* **1\_extract\_reddit\_data.py:** Parses submissions and comments from the provided Subreddits and saves them to a pandas DataFrame.
* **2\_recreate\_submissions.py:** Builds a graph to recreate submission discussion trees. We need this to compute structural features.
* **3\_filter\_comments.py:** Filter the submission DataFrame to only contain those submissions with at least 10 comments.
* **4\_filter\_graph.py:** Filter the submission graph to only contain those submissions with at least 10 comments.
* **5\_clean\_comments.py:** This file cleans the text of comments.
* **6\_detect\_language.py:** Detects the language of individual comments.
* **7\_compute\_features.py:** Computes all the features. This can take a very long time (depending on your hardware).
* **8\_conduct\_analysis.py:** With this file, you can conduct the different analyses presented in the paper. Just uncomment the respective lines in the main section of the code.
* **9\_conduct\_prediction.py:** This file conducts the prediction experiment. You need to alter the grid search manually by uncommenting the respective passage in the code.
