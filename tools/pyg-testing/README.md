# PyG Testing

Testing E2E pyg/examples models automatically. 

## Convert Model in source code level

When you want to profile some thing, you need to add a lot of duplicate code such as `start = time.start()`, `stop = ...`. 

We have a script `run_convert.py` to help you automatically insert these profiling code. 

### 1: general perf testing

Automatically insert argparse, e2e time logic.

Usage:
```
python run_convert.py \
    --pattern-dir ./codetrans_pattern/e2e
    --result-dir "your result code dir, please using an empty folder 
    --code-dir "PyG Examples folder(such as /pytorch_geometric/examples)  "
```

### 1: jittable perf testing

Automatically insert argparse, e2e time, gpu time and jittable logic.

Usage:
```
python run_convert.py \
    --pattern-dir ./codetrans_pattern/jit
    --result-dir "your result code dir, please using an empty folder 
    --code-dir "PyG Examples folder(such as /pytorch_geometric/examples)  "
```


See `/example/agnn_before.py` `/example/e2e.py` `/example/agnn_jit.py` to get more detail.



## Run E2E Testing 

Run E2E testing on PyG Models

```
python run_nsys.py \
    --warmup 3 --iter-time 2 --perf-time 2 \
    --log_dir ${E2E_LOG_Dir} \
    --run_dir ${E2E_Model_Dir} \
    --result_dir ${E2E_RES_Dir}
```
