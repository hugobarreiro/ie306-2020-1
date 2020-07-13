#!/bin/bash

python prova_01/src/02/script.py
python prova_01/src/03/script.py
cd prova_01/tex && pdflatex -output-directory=../out document
