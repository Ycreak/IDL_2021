#!/bin/bash
cd ..; 
python3 task1.py --img2text --create_model --evaluate --split 0.2 --confusion-matrix; 
python3 task1.py --img2text --create_model --evaluate --bidirectional --split 0.2 --confusion-matrix; 
