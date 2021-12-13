#!/bin/bash
cd ..; 
python3 task1.py --text2img --create_model --evaluate --split 0.2 --epochs 100; 
python3 task1.py --text2img --create_model --evaluate --bidirectional --split 0.2 --epochs 100; 
