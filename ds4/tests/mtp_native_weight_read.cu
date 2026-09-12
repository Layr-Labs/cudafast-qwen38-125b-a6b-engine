#include "ds4_cuda_qwen4exp.cuh"
/* Read-only test helper for the actual device-resolved weight mapping. */
extern "C" int mtp_native_read_weights(void *out,const void *map,unsigned long long bytes){
 const char *w=ds4_cuda_qwen4exp_weight_ptr(map,0,bytes,0,"native sort weight test");
 return w && cudaMemcpy(out,w,bytes,cudaMemcpyDeviceToHost)==cudaSuccess;
}
