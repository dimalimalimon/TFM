#!/bin/bash

python run_subtype.py E B_1000_trained > log_E.txt
python run_subtype.py A B_1000_trained > log_A.txt
python run_subtype.py G B_1000_trained > log_G.txt
