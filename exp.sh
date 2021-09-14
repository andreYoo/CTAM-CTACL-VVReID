#!/bin/bash
python tools/train.py --experiments experiments/veri776.yml --gpus 0,1 --suffix 'veri776'
python tools/train.py --experiments experiments/vveri901.yml --gpus 0,1 --suffix 'vveri901'
python tools/train.py --experiments experiments/veri-wild.yml --gpus 0,1 --suffix 'veri-wild'
