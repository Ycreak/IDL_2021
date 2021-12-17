#!/bin/bash
cd ..; 
python3 task1.py --text2text --create_model --split 0.05;
python3 task1.py --text2text --create_model --split 0.1; 
python3 task1.py --text2text --create_model --split 0.2; 
python3 task1.py --text2text --create_model --split 0.4; 
python3 task1.py --text2text --create_model --split 0.5; 
python3 task1.py --text2text --create_model --split 0.8;
python3 task1.py --text2text --create_model --split 0.9;
python3 task1.py --text2text --create_model --split 0.95;

python3 task1.py --text2text --create_model --split 0.05 --bidirectional;
python3 task1.py --text2text --create_model --split 0.1 --bidirectional; 
python3 task1.py --text2text --create_model --split 0.2 --bidirectional; 
python3 task1.py --text2text --create_model --split 0.4 --bidirectional; 
python3 task1.py --text2text --create_model --split 0.5 --bidirectional; 
python3 task1.py --text2text --create_model --split 0.8 --bidirectional;
python3 task1.py --text2text --create_model --split 0.9 --bidirectional;
python3 task1.py --text2text --create_model --split 0.95 --bidirectional;
