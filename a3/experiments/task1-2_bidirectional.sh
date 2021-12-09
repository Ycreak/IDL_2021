#!/bin/bash
cd ..; 
python3 task2.py --text2text --create_model --evaluate --split 0.2; 
python3 task2.py --text2text --create_model --evaluate --bidirectional --split 0.2; 
