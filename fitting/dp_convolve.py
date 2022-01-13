import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule
import time
from scipy import signal
import numpy as np
kernel2 = DynamicSourceModule("""
#define radius_x 50
#define radius_y 50
#define out_x 20
#define out_y 30
#include<stdio.h>
#include<math.h>
//#if __CUDA_ARCH__ < 600
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


__global__ void conv2d(double *input, double *image, double *dest,double mask[2*radius_y+1][2*radius_x+1],
                        int y_size,  int x_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Col_i = bx * blockDim.x + tx;
    int Row_i = by * blockDim.y + ty;

    double output[out_x * out_y];

    if (Row_i<= radius_y + out_y -1 && Row_i>=radius_y
        && Col_i<=radius_x + out_x -1 && Col_i>=radius_x){
        double p_value = 0.0;
        for (int i=0;i<2*radius_y+1;i++){
          for (int j=0;j<2*radius_x+1;j++){
            p_value += mask[i][j] * input[(Row_i-radius_y+i)*x_size+(Col_i-radius_x+j)];
              }
          }

        output[(Row_i-radius_y)*out_x+Col_i-radius_x]=p_value;
        //__syncthreads();
        //atomicAdd(dest,(output[(Row_i-radius_y)*out_x+Col_i-radius_x]
        //                -image[(Row_i-radius_y)*out_x+Col_i-radius_x]));
                        //*(output[(Row_i-radius_y)*out_x+Col_i-radius_x]
                        //-image[(Row_i-radius_y)*out_x+Col_i-radius_x])
                        //);
        atomicAdd((double*)dest,(double)(output[(Row_i-radius_y)*out_x+Col_i-radius_x]-image[(Row_i-radius_y)*out_x+Col_i-radius_x])
                                        *(output[(Row_i-radius_y)*out_x+Col_i-radius_x]-image[(Row_i-radius_y)*out_x+Col_i-radius_x]));
        //cudaDeviceSynchronize();
        //__syncthreads();
        }
    //__syncthreads();
 }

 __global__ void kernel_convolve(double* __restrict__ input, 
                                 double* __restrict__ image, 
                                 double *__restrict__ arr, 
                                 double mask[2*radius_y+1][2*radius_x+1],
                                 int y, int x, int len){
   int tx = threadIdx.x + blockIdx.x * blockDim.x;
   dim3 dimGrid(10,10,1);
   dim3 dimBlock(32,16,1);

   if (tx < len){
     cudaStream_t s;
     cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
     arr[tx] = 0;
     conv2d<<<dimGrid,dimBlock,0,s>>>(input, image, &arr[tx], 
                                    mask, y, x);
                   
     cudaDeviceSynchronize();
     __syncthreads();
   }

 }
""")
conv2d2 = kernel2.get_function('kernel_convolve')


def gen_input(x,mask_row,mask_col):
    rx = (mask_col-1)//2
    ry = (mask_row-1)//2
    arr_temp = np.zeros((ry,x.shape[1]))
    x1 = np.vstack((arr_temp,x))
    x1 = np.vstack((x1,arr_temp))
    arr_temp2 = np.zeros((x1.shape[0],rx))
    x1 = np.hstack((arr_temp2,x1))
    x1 = np.hstack((x1,arr_temp2))
    return x1

x1 = np.arange(1,30*20+1).reshape(30,20).astype(np.float64)
rx,ry = 50,50
mask = np.arange((2*rx+1)*(2*ry+1)).reshape(2*ry+1,2*rx+1).astype(np.float64)
mask_converted = mask[::-1,::-1].astype(np.float64)
mask_gpu = gpuarray.to_gpu(mask_converted)

x = gen_input(x1,mask.shape[0],mask.shape[1])

x_gpu = gpuarray.to_gpu(x)
out_gpu = gpuarray.zeros(x1.shape,np.float64)

d_gpu = gpuarray.zeros(2000,np.float64)
#image = np.ones(x_gpu.shape,np.float64)*8
image = np.random.rand(x1.shape[0],x1.shape[1])
image_gpu = gpuarray.to_gpu(image)

t1 = time.time()
conv2d2(x_gpu,image_gpu,d_gpu,mask_gpu,np.int32(x.shape[0]),np.int32(x.shape[1]),np.int32(2000),grid=(100,1),block=(32,1,1))
t2 = time.time()
print(t2-t1)

#result = out_gpu.get()
#imshow(result,aspect= 'auto');colorbar()
d_result = d_gpu.get()

t1 = time.time()
ref = signal.convolve2d(x1,mask,'same')
t2 = time.time()
print(t2-t1)

#sum(result),
#sum(ref**2),(d_result)
#allclose(d_result[0],sum(ref**2))
print(np.allclose(d_result[1500],np.sum((ref-image)**2)))
