#! /bin/bash

nvidia-smi --query-supported-clocks=gr,mem --format=csv
nvidia-smi -pm 1
nvidia-smi -lgc 1200,1200
nvidia-smi -lmc 1512,1512