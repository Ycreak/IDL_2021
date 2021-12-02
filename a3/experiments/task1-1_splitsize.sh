#!/bin/bash
cd ..; 
python3 task1.py --text2text --create_model --split 0.05;
python3 task1.py --text2text --create_model --split 0.1; 
python3 task1.py --text2text --create_model --split 0.2; 
python3 task1.py --text2text --create_model --split 0.4; 
python3 task1.py --text2text --create_model --split 0.5; 
python3 task1.py --text2text --create_model --split 0.8;