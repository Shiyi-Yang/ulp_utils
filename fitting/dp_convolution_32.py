def conv(image,psf):
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pycuda.compiler import DynamicSourceModule    
    import numpy as np
    
    def gen_input(x,mask_row,mask_col):
        rx = (mask_col-1)//2
        ry = (mask_row-1)//2
        arr_temp = np.zeros((ry,x.shape[1]))
        x1 = np.vstack((arr_temp,x))
        x1 = np.vstack((x1,arr_temp))
        arr_temp2 = np.zeros((x1.shape[0],rx))
        x1 = np.hstack((arr_temp2,x1))
        x1 = np.hstack((x1,arr_temp2))
        x1 = x1.astype(np.float32)
        return x1
    
    def gen_mask(psf):

        arr = np.zeros((psf.shape[0],1))
        psf_conv = np.hstack((psf,arr))
        arr = np.zeros(psf_conv.shape[1])
        psf_conv = np.vstack((psf_conv,arr))

        normalize = lambda x:x/np.amax(x)
        psf_conv = normalize(psf_conv)
        mask = psf_conv[::-1,::-1].astype(np.float32)

        return mask
    
    kernel = DynamicSourceModule("""
    #define radius_x 256
    #define radius_y 125
    #define out_x 128
    #define out_y 300
    
    #include<stdio.h>
    #include<math.h>


    __global__ void conv2d(float *input, float *dest,float mask[2*radius_y+1][2*radius_x+1], int y_size, int x_size) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        int Col_i = bx * blockDim.x + tx;
        int Row_i = by * blockDim.y + ty;

        if (Row_i<= radius_y + out_y -1 && Row_i>=radius_y
            && Col_i<=radius_x + out_x -1 && Col_i>=radius_x){
            float p_value = 0.0;
            for (int i=0;i<2*radius_y+1;i++){
              for (int j=0;j<2*radius_x+1;j++){
                p_value += mask[i][j] * input[(Row_i-radius_y+i)*x_size+(Col_i-radius_x+j)];
                  }
              }
            dest[(Row_i-radius_y)*out_x+Col_i-radius_x]=p_value;
           
            }
        
     }

     __global__ void kernel_convolve(float* __restrict__ input, 
                                     float *__restrict__ arr, 
                                     float mask[2*radius_y+1][2*radius_x+1],
                                     int y, int x, int len){
       int tx = threadIdx.x + blockIdx.x * blockDim.x;
       dim3 dimGrid(ceil(float(x)/32),ceil(float(y)/16),1);
       dim3 dimBlock(32,16,1);

       if (tx < len){
         cudaStream_t s;
         cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

         conv2d<<<dimGrid,dimBlock,0,s>>>(&input[tx*y*x], &arr[tx*out_x*out_y], 
                                        mask, y, x);

         cudaDeviceSynchronize();
         __syncthreads();
       }

     }
    """)
    conv2d = kernel.get_function('kernel_convolve')

   
    mask = gen_mask(psf)
    mask_gpu = gpuarray.to_gpu(mask)
    temp = gen_input(image[1],mask.shape[0],mask.shape[1])
    temp_y, temp_x = temp.shape[0],temp.shape[1]
    image_pad = np.zeros((image.shape[0],temp_y,temp_x)).astype(np.float32)
    for i in range(image.shape[0]):
        image_pad[i] = gen_input(image[i],mask.shape[0],mask.shape[1])
    
    x_gpu = gpuarray.to_gpu(image_pad)
   
    dest_gpu = gpuarray.zeros((image.shape),np.float32)
  
    conv2d(x_gpu,dest_gpu,mask_gpu,np.int32(temp_y),np.int32(temp_x),\
            np.int32(dest_gpu.shape[0]),grid=(dest_gpu.shape[0]//32+1,1,1),block=(32,1,1))
    #from pycuda.autoinit import context
    #context.synchronize()
    out = dest_gpu.get()

    x_gpu.gpudata.free()
    dest_gpu.gpudata.free()
    mask_gpu.gpudata.free()

    return out

def gen_mask(psf):

    arr = np.zeros((psf.shape[0],1))
    psf_conv = np.hstack((psf,arr))
    arr = np.zeros(psf_conv.shape[1])
    psf_conv = np.vstack((psf_conv,arr))

    normalize = lambda x:x/np.amax(x)
    psf_conv = normalize(psf_conv)
    psf_conv = psf_conv.astype(np.float32)
    return psf_conv

if __name__== "__main__":
    import numpy as np
    import time
    import numpy.linalg as la
    input_arr = np.random.rand(10,800,128).astype(np.float32)
    #for i in range(input_arr.shape[0]):
    #    input_arr[i] = np.arange(i,30*20+i).reshape(30,20).astype(np.float64)
    rx,ry = 256,500
    #mask = np.arange((2*rx+1)*(2*ry+1)).reshape(2*ry+1,2*rx+1).astype(np.float64)
    mask = np.random.rand(1000,512).astype(np.float32)
    mask_m = gen_mask(mask)
    
    #image = np.random.rand(input_arr.shape[1],input_arr.shape[2]).astype(np.float32)
    print("starting")
    t1 = time.time()
    out = conv(input_arr,mask)
    t2 = time.time()
    
    print("time",t2-t1)
    from scipy import signal
    ref_arr = []
    index = 0
    for i in range(index,index + 5):
        ref = signal.convolve2d(input_arr[i],mask_m,'same')
        print(np.allclose(ref,out[i],rtol=1e-5))
        #print(ref)
        #print("")
        #print(out[0])
        #ref_arr += [ref]
    #for i in range(len(ref_arr)):
        #print (i,(ref_arr[i],out[i+index]))
        #print (i,np.allclose(ref_arr[i],out[i+index]))
    #    print(out[i,0])
    #    print("")
    #    print(ref)
    #print(ref.dtype)