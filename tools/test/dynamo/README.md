
## Run Dynamo & Get NVFuser Result
```
PYTORCH_JIT_LOG_LEVEL="graph_fuser" python dynamo_autoencoder.py > dynamo_autoencoder_graph.log 2>&1
```

NVFuser will fuse kernel into cudaFusionGroup, for example:
```
[DUMP graph_fuser.cpp:2506] with prim::CudaFusionGroup_0 = graph(%2 : Long(2526, strides=[1], requires_grad=0, device=cuda:0),
[DUMP graph_fuser.cpp:2506]       %3 : Long(2526, strides=[1], requires_grad=0, device=cuda:0)):
[DUMP graph_fuser.cpp:2506]   %ne.1 : Bool(2526, strides=[1], requires_grad=0, device=cuda:0) = aten::ne(%2, %3) # <eval_with_key>.3:7:0
[DUMP graph_fuser.cpp:2506]   %1 : Bool(2526, strides=[1], requires_grad=0, device=cuda:0) = aten::bitwise_not(%ne.1) # <eval_with_key>.3:12:0
[DUMP graph_fuser.cpp:2506]   return (%1, %ne.1)
```