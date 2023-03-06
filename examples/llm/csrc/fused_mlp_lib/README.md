This CUDA extension comes from HazyResearch: https://github.com/HazyResearch/flash-attention/blob/main/csrc/fused_dense_lib/


This version supports tensor parallelism, while the ../fused_dense_lib/ folder does not.

Their README reads:
This CUDA extension implements fused matmul + bias (forward and backward), and fused matmul + bias + gelu
(forward and backward), adapted from Apex's
[FusedDense](https://github.com/NVIDIA/apex/tree/master/apex/fused_dense).
We make it work for bfloat16.

For best performance, you should use CUDA >= 11.8. CuBLAS versions before
this doesn't have the best matmul + bias + gelu performance for bfloat16.

It has only been tested on A100s.

```sh
cd csrc/fused_dense_lib && pip install .
```
