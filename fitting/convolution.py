def conv_2d(image,psf):
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    kernel = SourceModule("""
    #define radius_x 8192
    #define radius_y 125
    #define out_x 8192
    #define out_y 751
    #include<stdio.h>
    __global__ void conv2d(double *input, double *output,double mask[2*radius_y+1][2*radius_x+1], int y_size,  int x_size) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        int Col_i = bx * blockDim.x + tx;
        int Row_i = by * blockDim.y + ty;

        if (Row_i<= radius_y + out_y -1 && Row_i>=radius_y
            && Col_i<=radius_x + out_x -1 && Col_i>=radius_x){
            double p_value = 0.0f;
            for (int i=0;i<2*radius_y+1;i++){
              for (int j=0;j<2*radius_x+1;j++){
                p_value += mask[i][j] * input[(Row_i-radius_y+i)*x_size+(Col_i-radius_x+j)];
                  }
              }

            output[(Row_i-radius_y)*out_x+Col_i-radius_x]=p_value;

            }


      __syncthreads();

     }
    """)
    conv2d = kernel.get_function('conv2d')

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

    def gen_mask(psf):

        arr = np.zeros((psf.shape[0],1))
        psf_conv = np.hstack((psf,arr))
        arr = np.zeros(psf_conv.shape[1])
        psf_conv = np.vstack((psf_conv,arr))

        normalize = lambda x:x/np.amax(x)
        psf_conv = normalize(psf_conv)
        mask = psf_conv[::-1,::-1].astype(np.float64)

        return mask

    mask = gen_mask(psf)
    mask_gpu = gpuarray.to_gpu(mask)
    x = gen_input(image,mask.shape[0],mask.shape[1])
    x_gpu = gpuarray.to_gpu(x)
    out_gpu = gpuarray.zeros(image.shape,np.float64)
    ##out_gpu = gpuarray.zeros((np.int(image.shape[0]),np.int(image.shape[1]//2)),np.float64)
    y_size,x_size=x.shape[0],x.shape[1]
    conv2d(x_gpu,out_gpu,mask_gpu,np.int32(y_size),np.int32(x_size),grid=(x_size//32+1,y_size//16+1),block=(32,16,1))
    result = out_gpu.get()

    x_gpu.gpudata.free()
    out_gpu.gpudata.free()
    mask_gpu.gpudata.free()

    return result


def func_conv2d_fw(image,psf):
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    kernel = SourceModule("""
    #define radius_x 8192
    #define radius_y 500
    #define out_x 16384
    #define out_y 10
    #include<stdio.h>
    __global__ void conv2d(double *input, double *output,double mask[2*radius_y+1][2*radius_x+1],
                            int y_size,  int x_size) {

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        int Col_i = bx * blockDim.x + tx;
        int Row_i = by * blockDim.y + ty;

        if (Row_i<= radius_y + out_y -1 && Row_i>=radius_y
            && Col_i<=radius_x + out_x -1 && Col_i>=radius_x + radius_x){
            double p_value = 0.0f;
            for (int i=0;i<2*radius_y+1;i++){
              for (int j=0;j<2*radius_x+1;j++){
                p_value += mask[i][j] * input[(Row_i-radius_y+i)*x_size+(Col_i-radius_x+j)];
                  }
              }

            output[(Row_i-radius_y)*out_x/2+Col_i-2*radius_x]=p_value;

            }


      __syncthreads();

     }
    """)
    conv2d = kernel.get_function('conv2d')

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

    def gen_mask(psf):

        arr = np.zeros((psf.shape[0],1))
        psf_conv = np.hstack((psf,arr))
        arr = np.zeros(psf_conv.shape[1])
        psf_conv = np.vstack((psf_conv,arr))

        normalize = lambda x:x/np.amax(x)
        psf_conv = normalize(psf_conv)
        mask = psf_conv[::-1,::-1].astype(np.float64)

        return mask

    mask = gen_mask(psf)
    mask_gpu = gpuarray.to_gpu(mask)
    x = gen_input(image,mask.shape[0],mask.shape[1])
    x_gpu = gpuarray.to_gpu(x)
    ##out_gpu = gpuarray.zeros(image.shape,np.float64)
    out_gpu = gpuarray.zeros((np.int(image.shape[0]),np.int(image.shape[1]//2)),np.float64)
    y_size,x_size=x.shape[0],x.shape[1]
    conv2d(x_gpu,out_gpu,mask_gpu,np.int32(y_size),np.int32(x_size),grid=(x_size//32+1,y_size//16+1),block=(32,16,1))
    result = out_gpu.get()

    x_gpu.gpudata.free()
    out_gpu.gpudata.free()
    mask_gpu.gpudata.free()

    return result
if __name__ == "__main__":
    import numpy as np
    
    data = np.load('/home/yang158/notebook/ISR_model/task_gan.npz')
    clean = data['clean']
    freq = data['freq']
    height=data['height']
    data = np.load('/home/yang158/notebook/ISR_model/test_fitting_data.npz')
    psf = data['psf']
    psf = psf.reshape(4,psf.shape[0]//4,psf.shape[1]).mean(axis=0)
    dirty = conv_2d(clean,psf)
    np.savez('/home/yang158/notebook/ISR_model/task_gan_conv',spec = dirty)