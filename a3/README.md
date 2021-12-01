
<!-- TASK 1 -->
To create the dataset
$ python3 task1.py --create_dataset

To create the model and save it to disk
$ python3 task1.py --create_model

To run the split size experiment from 1.1
$ python3 task1.py --text2text --create_model --split 0.05; python3 task1.py --text2text --create_model --split 0.1; python3 task1.py --text2text --create_model --split 0.2; python3 task1.py --text2text --create_model --split 0.4; python3 task1.py --text2text --create_model --split 0.5; python3 task1.py --text2text --create_model --split 0.8