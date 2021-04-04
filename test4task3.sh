#!/usr/bin/env bash

cd bert4task1
python test.py ./experiments ../data/task3-dev.json ../data/task3-dev-result-phase1.json
cd ../bert4task2
python test.py ./experiments ../data/task3-dev-result-phase1.json ../data/task3-dev-result.json

