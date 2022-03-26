#!/bin/bash

for n in 0.02 0.04 0.06 0.08 0.10 0.12
do
    python test_classification.py --noise $n
done