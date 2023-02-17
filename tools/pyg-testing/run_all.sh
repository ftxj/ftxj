#!/bin/bash

# TODO: lock GPU frequency, bind cpu core 

PYG_Model_Dir="/workspace/env/pytorch_geometric/examples"

E2E_Model_Dir="/workspace/test/e2e_dir"
E2E_LOG_Dir="/workspace/test/e2e_log_dir"
E2E_RES_Dir="/workspace/test/e2e_res_dir"

JIT_Model_Dir="/workspace/test/jit_dir"
JIT_LOG_Dir="/workspace/test/jit_log_dir"
JIT_RES_Dir="/workspace/test/jit_res_dir"

# generate e2e test models
python run_convert.py \
    --pattern-dir ./codetrans_pattern/nojittable.json \
    --code-dir ${PYG_Model_Dir} \
    --result-dir ${E2E_Model_Dir}

# run e2e test
python run_e2e.py \
    --warmup 3 --iter-time 2 --perf-time 2 \
    --log_dir ${E2E_LOG_Dir} \
    --run_dir ${E2E_Model_Dir} \
    --result_dir ${E2E_RES_Dir}

