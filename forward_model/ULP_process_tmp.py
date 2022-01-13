#!/usr/bin/env python
#PBS -l nodes=2:ppn=6,walltime=3600 

#import matplotlib
#import matplotlib.pyplot as plt
import time
import glob
import sys,os
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.fft as cufft
from pycuda.compiler import SourceModule
import bz2,h5py
def process_spec(yyyy,mm,dd,hh,mmin,ss):
    from time import gmtime,strftime
    from calendar import timegm
    from glob import glob1
    import numpy as np
    import sys,os
    nyear = int(yyyy)
    nmonth = int(mm)
    nday = int(dd)
    nhh = int(hh)
    nmm = int(mmin)
    nss = int(ss)
    
    ##Create folders to saving
    specfolder_year = '/home/yang158/processed/%.4d/'%nyear
    specfolder = '/home/yang158/processed/%.4d/%.4d_%.2d_%.2dT%.2d_%.2d/'%(nyear,nyear,nmonth,nday,nhh,nmm)

    if not os.path.exists(specfolder_year):
        os.makedirs(specfolder_year)
    if not os.path.exists(specfolder):
        os.makedirs(specfolder)

    ##Reading data in each file
    def read_file_data(fname,verbose=False):
        #import h5py
        fp =h5py.File(fname,"r")
        data = fp.get('rf_data')
        if verbose:
            print (data.dtype)
        ndata = data['r'].squeeze() + 1j * data['i'].squeeze()
        return ndata.astype("complex64")
    ##Reading bz_file
    def read_file_data_bz(fname):
        #import h5py
        temp_file=open('/dev/shm/temp.h5','wb')
        temp_file.write(fname)
        temp_file.close()
        fp=h5py.File('/dev/shm/temp.h5','r')
        data = fp.get('rf_data')

        ndata = data['r'].squeeze() + 1j * data['i'].squeeze()
        return ndata.astype("complex64")
    #Modified code
    def Detect_IPP_Tx(ndata0,verbose=False):
        decim=100
        pwr100 = (abs(ndata0)**2).reshape(ndata0.shape[0]//decim,decim).sum(1)
        derv100 = pwr100[1:]-pwr100[:-1]
        thresh1 = 1e8
        thresh2 = -1e8/1.5
        curr_test = 0

        ###Detect the starting pulse    
        try:    
            curr_test += np.nonzero(derv100[curr_test:]>thresh1)[0][0]
            TxUp = np.nonzero(derv100[curr_test:]<thresh2)[0][0]
        except:
            TxWs = []
            IPPs_start = []
            TxWs = np.array(TxWs)
            IPPs_start = np.array(IPPs_start)

            if verbose:
                print ("No Pulses detected")

            return TxWs, IPPs_start
        while TxUp < 100:
            if verbose:
                print ("false pulse")
            curr_test += TxUp

            try:
                curr_test += np.nonzero(derv100[curr_test:] > thresh1)[0][0]
                TxUp = np.nonzero(derv100[curr_test:]<thresh2)[0][0]
                if verbose:
                    print (curr_test,  TxUp)
            except:
                TxWs = []
                IPPs_start = []
                if verbose:
                    print ("No Pulses detected")
                TxWs = np.array(TxWs)
                IPPs_start = np.array(IPPs_start)

                return TxWs, IPPs_start
        IPPs_start = [curr_test * decim]   # Start of first Tx
        TxWs = [TxUp * decim]

        ##10 secs clp 20 secs ULP 10ms ipp
        if TxUp == 125:
            for num_iteration in range(99):

                #curr_test += (i+1) * 2500 ##10ms Ipp
                try:
                    TxUp = np.nonzero(derv100[curr_test + 2500:] < thresh2)[0][0]
                except:
                    break
                if TxUp == 125:
                    curr_test += 2500
                    IPPs_start += [curr_test * decim]
                    TxWs += [TxUp * decim]
                else:
                    break
                #elif TxUp == 250:
                #    curr_test += 5000
                #    IPPs_start += [curr_test * decim]
                #    TxWs += [TxUp * decim]
                #    break

        if TxUp == 110:
            for num_iteration in range(99):

                #curr_test += 2500  ##10ms IPP
                try:
                    TxUp = np.nonzero(derv100[curr_test:]<thresh2)[0][0]
                except:
                    break
                if TxUp == 110:
                    curr_test += 2500
                    IPPs_start += [curr_test* decim]
                    TxWs += [TxUp * decim]
                else:
                    break
        '''
        if TxUp == 250: ##50 pulses, 20ms IPP
            #curr_test += 5000 
            for num_iteration in range(99):

                try:
                    TxUp = np.nonzero(derv100[curr_test + 5000:]<thresh2)[0][0]
                except:
                    break
                if TxUp == 250:
                    curr_test += 5000
                    IPPs_start += [curr_test * decim]
                    TxWs += [TxUp * decim]

                elif TxUp == 125:
                    curr_test += 2500
                    IPPs_start += [curr_test * decim]
                    TxWs += [TxUp * decim]

                    break

        '''
        IPPs_start = np.array(IPPs_start)
        TxWs = np.array(TxWs)

        startTx = IPPs_start[0]
        startscan = max(0,startTx-100) # in case pulse is close to the start of the file
        maxTxval = max(abs(ndata0[startscan:startscan+400]))
        findresult = np.nonzero(abs(ndata0[startscan:startscan+400])>maxTxval/2)[0]
        if len(findresult)>0:
            newstartTx = startscan + findresult[0]
            correction = newstartTx - startTx
        else:
            correction = 0

        IPPs_start += correction
        return TxWs,IPPs_start   

  
           
    def get_delta_f_ph(data):
        x = np.fft.fftshift(abs(np.fft.fft(data))**2)
        omega = np.argmax(x)

        return omega - len(data)//2
 
    
    ##Generate the PSF
    def psf_func(txdata0):
        N = 12500
        t = np.arange(N)
        #TxWidth, IPPs_start = Detect_IPP_Tx(data,verbose=False)
        #txdata0 = data[IPPs_start[0]:IPPs_start[0] + TxWidth[0]]
        omega = get_delta_f_ph(txdata0)
        if omega<0:
            modsignal = np.exp(1j*t*62.5/2*2*np.pi/N)
        if omega>0:
            modsignal = np.exp(-1j*t*62.5/2*2*np.pi/N)
        basebandTx = txdata0 * modsignal
        dest_cpu = np.zeros([2*N,N+3884],np.complex64)
        dest_output = np.empty([2*N,N+3884],np.complex64)
        dest = np.empty(dest_cpu.shape).astype(np.complex64)

        for i in range(1,N):
            dest_cpu[i,0:i] = basebandTx[N-i:] ##leftshift
        for i in range(N):
            dest_cpu[i+N,:N-i] = basebandTx[:N-i]##rightshift

        nFFT = 16384
        batch = 2**13

        for i in range(dest_cpu.shape[0]):
            dest[i,:] = np.fft.fft(dest_cpu[i,:])
        

        dest2 = abs(dest)**2
        psf = dest2.reshape(1000,25,16384).mean(axis=1)
        psf = np.fft.fftshift(psf,axes=1)
        ref = np.average(psf[:50,:500])
        psf_ref = np.copy(psf)
        for i in range(psf.shape[0]):
            reference = np.amax(psf[i,:])
            for j in range(8192,8292+300):
                if 10*np.log10(psf[i,j])<10*np.log10(reference)-10:
                    psf_ref[i,j:] = 0
                    psf_ref[i,:2*8192-j] = 0

                    break
        return psf_ref        
    
    nFFT = 16384
    batch = 7864
    #Rearrange the one dimension array into a 20000*11000 matrix
    plan2 = cufft.Plan(shape = (nFFT,), in_dtype=np.complex64, out_dtype=np.complex64,
                     batch = batch, stream = None, mode = 1,
                     inembed = np.array([nFFT],dtype = int),
                     istride = 1,
                     idist =nFFT,
                     onembed = np.array([nFFT],dtype = int),
                     ostride =1,
                     odist = nFFT)
    kernel = SourceModule("""
    #include <stdio.h>
    #include <complex.h>
    __global__ void rearrange(float2 *a, float2 *dest)
    {
    //int Rows = blockIdx.y*blockDim.y + threadIdx.y;
    int Cols = blockIdx.x*blockDim.x + threadIdx.x;
    int batch = 7864;
    int nFFT = 12500;
    if(Cols < nFFT)
    {
        for (int i = 0; i < batch; i++)
        dest[Cols + nFFT * i] = a[Cols + i * 25];
    }

    }
    """)
    rearrange = kernel.get_function('rearrange')
    #Getting the magnitude of the matrix and add them up
    kernel1 = SourceModule("""
    #include <stdio.h>
    #include<complex.h>
    __global__ void power(float2 *a, float *dest)
    {

    //Thread index
    int nFFT = 16384;
    int batch = 7864;
    const int Rows = blockIdx.y * blockDim.y + threadIdx.y;
    const int Cols = blockIdx.x * blockDim.x + threadIdx.x;
    int a_index = Cols + nFFT * Rows;
    float reala,imaga;
    if (Rows < batch && Cols < nFFT)
    {
        reala = a[a_index].x;
        imaga = a[a_index].y;
        __syncthreads();
        dest[a_index] += reala * reala + imaga * imaga; 

    }
    }
    """)
    power = kernel1.get_function('power')
    
    ##Multiply the samples with corresponding code
    kernel2 = SourceModule("""
    __global__ void shifting(float2 *a, float2 *cod)
    {
    int Rows = blockIdx.y*blockDim.y + threadIdx.y;
    int Cols = blockIdx.x*blockDim.x + threadIdx.x;
    float real;
    float imag;
    int nFFT = 12500;
    int batch = 7864;
    if (Rows < batch && Cols < nFFT){
        real = a[Cols + Rows * nFFT].x * cod[Cols].x - a[Cols + Rows * nFFT].y * cod[Cols].y;
        imag = a[Cols + Rows * nFFT].y * cod[Cols].x + a[Cols + Rows * nFFT].x * cod[Cols].y;
        __syncthreads();
        a[Cols + Rows * nFFT].x = real;
        a[Cols + Rows * nFFT].y = imag;
    }


    }
    """)
    shifting = kernel2.get_function('shifting')
    
    def decimation(spec):
    #    spectrum = spec[:196600,:].reshape(7864,25,16384).mean(axis=1)
        spectrum = np.fft.fftshift(spec,axes=1)[500:,:] 

        return spectrum

    def band_pass_filter(spec):
        fnoise = spec[7000:,:].mean(0)
        spec = spec/fnoise

        return spec

    def pulse_process(numFile,length): ##10 sec ULP processing
        counter = 0
        nFFT = 16384
        batch = 7864
        ## Store data in CPU
        cpu_dest = np.zeros([batch,nFFT],np.float32)  
        ## Store data after shifting
        dest_temp = gpuarray.empty([batch,nFFT-3884],np.complex64)
        ## Store the zero-padding data
        gpu_zero_padding = gpuarray.zeros([batch,nFFT],np.complex64)
        #Store FFT value
        dest_temp_gpu = gpuarray.empty((batch,nFFT),np.complex64)
        #store the power value
        dest_gpu = gpuarray.zeros([batch,nFFT],np.float32)

        N = 12500
        t = np.arange(N)

        while (counter < 19 and numFile < length):
            fph5=bz2.BZ2File(os.path.join(filespath,flist[numFile]),'rb').read() 
            ndata = read_file_data_bz(fph5)
            #ndata = read_file_data(os.path.join(filespath,flist[numFile]))
            TxWidth, IPPs_start = Detect_IPP_Tx(ndata,False)
            numFile += 1
            #if (len(TxWidth_0) != 100 or len(IPPs_start_0) != 100):
                #numFile += 1   
            if (len(TxWidth) == 100 and len(IPPs_start) == 100 and all(IPPs_start[1:]-IPPs_start[:-1]==250000) and all(TxWidth==12500)):
                counter += 1
                x_gpu = gpuarray.to_gpu(ndata)
                test_pulse = ndata[IPPs_start[0]:IPPs_start[0] + 12500]
                omega = get_delta_f_ph(test_pulse)
                modsignal1 = np.exp(-1j*t*62.5/2*2*np.pi/N)
                modsignal2 = np.exp(1j*t*62.5/2*2*np.pi/N)
                modsignal1_gpu = gpuarray.to_gpu(modsignal1).astype(np.complex64)
                modsignal2_gpu = gpuarray.to_gpu(modsignal2).astype(np.complex64)

                if omega > 0:
                    for numIPPs in range(0,len(IPPs_start)-1,2):
                        #shifting              
                        rearrange(x_gpu[IPPs_start[numIPPs]:IPPs_start[numIPPs] + 25*batch + 12500 - 25],dest_temp,grid = (13,1),block = (1024,1,1))
                        shifting(dest_temp,modsignal1_gpu,grid = (391,246),block = (32,32,1))
                        #zero-padding
                        gpu_zero_padding[:,:dest_temp.shape[1]] = dest_temp
                        #Generating spectrum
                        cufft.fft(gpu_zero_padding,dest_temp_gpu,plan2) 
                        power(dest_temp_gpu,dest_gpu,grid = (512,246),block = (32,32,1))
                    for numIPPs in range(1,len(IPPs_start)-1,2):
                        #shifting                
                        rearrange(x_gpu[IPPs_start[numIPPs]:IPPs_start[numIPPs] + 25*batch + 12500 - 25],dest_temp,grid = (13,1),block = (1024,1,1))

                        shifting(dest_temp,modsignal2_gpu,grid = (391,246),block = (32,32,1))
                        #zero-padding
                        gpu_zero_padding[:,:dest_temp.shape[1]] = dest_temp
                        #Generating spectrum
                        cufft.fft(gpu_zero_padding,dest_temp_gpu,plan2) 
                        power(dest_temp_gpu,dest_gpu,grid = (512,246),block = (32,32,1))  
                if omega < 0:
                    for numIPPs in range(0,len(IPPs_start)-1,2):
                        #shifting                
                        rearrange(x_gpu[IPPs_start[numIPPs]:IPPs_start[numIPPs] + 25*batch + 12500 - 25],dest_temp,grid = (13,1),block = (1024,1,1))
                        #rearrange(x_gpu[IPPs_start[numIPPs]:IPPs_start[numIPPs] + 10*batch + 12500 - 10],dest_temp,grid = (13,1),block = (1024,1,1))

                        shifting(dest_temp,modsignal2_gpu,grid = (391,246),block = (32,32,1))
                        #zero-padding
                        gpu_zero_padding[:,:dest_temp.shape[1]] = dest_temp
                        #Generating spectrum
                        cufft.fft(gpu_zero_padding,dest_temp_gpu,plan2) 
                        power(dest_temp_gpu,dest_gpu,grid = (512,246),block = (32,32,1))
                    for numIPPs in range(1,len(IPPs_start)-1,2):
                        #shifting                
                        rearrange(x_gpu[IPPs_start[numIPPs]:IPPs_start[numIPPs] + 25*batch + 12500 - 25],dest_temp,grid = (13,1),block = (1024,1,1))
                        #rearrange(x_gpu[IPPs_start[numIPPs]:IPPs_start[numIPPs] + 25*batch + 12500 - 25],dest_temp,grid = (13,1),block = (1024,1,1))

                        shifting(dest_temp,modsignal1_gpu,grid = (391,246),block = (32,32,1))
                        #zero-padding
                        gpu_zero_padding[:,:dest_temp.shape[1]] = dest_temp
                        #Generating spectrum
                        cufft.fft(gpu_zero_padding,dest_temp_gpu,plan2) 
                        power(dest_temp_gpu,dest_gpu,grid = (512,246),block = (32,32,1))    

        cpu_dest = dest_gpu.get()                
        spectrum = decimation(cpu_dest)
        spec = band_pass_filter(spectrum)
        ##Free GPU memeory
        dest_temp.gpudata.free()
        gpu_zero_padding.gpudata.free()
        dest_gpu.gpudata.free()
        dest_temp_gpu.gpudata.free()
        x_gpu.gpudata.free()
        modsignal1_gpu.gpudata.free()
        modsignal2_gpu.gpudata.free()
    
        return spec
    
    
    
    def overlap(dirty,psf,mx,my):
        radius_x = int(psf.shape[1]/2)
        radius_y = int(psf.shape[0]/2)
        dxbegin = 0
        dxend = dirty.shape[1]
        pxbegin = radius_x - mx 
        pxend = pxbegin + dirty.shape[1]
        if my - radius_y < 0:
            dybegin = 0
            dyend = my + radius_y
            pybegin= radius_y - my
            pyend = 2 * radius_y
        elif my + radius_y > dirty.shape[0]:
            dybegin = my - radius_y
            dyend = dirty.shape[0]
            pybegin = 0
            pyend = radius_y - my + dirty.shape[0]

        #elif my-psf.shape[1]/2>=0 and my+psf.shape[1]/2<dirty.shape[1]:
        elif my + radius_y <= dirty.shape[0] and my - radius_y >= 0:
            dybegin = my - radius_y
            dyend = my + radius_y
            pybegin = 0
            pyend = 2 * radius_y


        return (int(dxbegin), int(dxend), int(dybegin), int(dyend)), (int(pxbegin), int(pxend), int(pybegin), int(pyend))
        
    def hogbom(dirty,psf,gain):
        #frac = 0
        res = np.array(dirty)
        cleaned = np.zeros(dirty.shape)
        point = np.ones(psf.shape)*0

        sigma_x = 20
        sigma_y = 10
        x = np.linspace(-100,100,201)
        y = np.linspace(-100,100,201)
        x,y = np.meshgrid(x,y)
        z = 1/np.sqrt(2*np.pi*sigma_x**2)*np.exp(-x**2/2/sigma_x**2) * 1./np.sqrt(2*np.pi*sigma_y**2)*np.exp(-y**2/2/sigma_y**2)
        z = z/np.amax(z)

        point[psf.shape[0]//2-100:psf.shape[0]//2+101,psf.shape[1]//2-100:psf.shape[1]//2+101] = z    

        for i in range(30000):
            my, mx=np.unravel_index(np.argmax(res), res.shape)
            mval= res[my, mx] * gain
            d_index, p_index = overlap(res,psf,mx,my)
            res[d_index[2]:d_index[3],d_index[0]:d_index[1]] -= psf[p_index[2]:p_index[3],p_index[0]:p_index[1]] * gain
            cleaned[d_index[2]:d_index[3],d_index[0]:d_index[1]] += mval*point[p_index[2]:p_index[3],p_index[0]:p_index[1]]

            ##Condition
            #if amax(res)<0.02:
            if np.amax(res)<0.002 or np.amax(res)<(dirty[4000:,:]).max():
            #if frac < 0.02
                break
        restored = cleaned + res
        fnoise = restored[4000:,:].mean(0)
        ref_spec = restored - fnoise

        return ref_spec
    
    
    
    def spline_fit(spec1,spec2):
        from scipy.interpolate import UnivariateSpline as spline

        x = np.linspace(75,689.4,4096)
        index = 2048
        y,y2 = [],[]
        for i in range(index):
            y.append(np.argmax(spec1[i,:]))
            y2.append(np.argmax(spec2[i,:]))
        
        y = 1.0*25*(np.array(y)+3184)/16384-12.5
        y2 = 1.0*25*(np.array(y2)+9104)/16384-12.5

        result = spline(x[:index],y)
        result.set_smoothing_factor(10)

        result2 = spline(x[:index],y2)
        result2.set_smoothing_factor(10)

        return x,y,y2,result(x),result2(x)           
    
    
    
    
    ##Main Program
    filespath = dpath+syear_arr[i]+'-'+smonth_arr[i]+'-'+sday_arr[i]+'T'+shh_arr[i]+'-'+smm_arr[i]+'-'+sss_arr[i]
    flist = sorted(glob.glob1(filespath,"*.bz2"))
    file_length = len(flist)
    #file_length = 20
    numFile = 0
    while(numFile + 19 < file_length):
        fph5=bz2.BZ2File(os.path.join(filespath,flist[numFile]),'rb').read() 
        ndatai = read_file_data_bz(fph5)
        #ndatai = read_file_data(os.path.join(filespath,flist[numFile]))
        
        TxWidth_0, IPPs_start_0 = Detect_IPP_Tx(ndatai,False)
        print ("processing the file:",numFile)
        #sys.stdout.flush()
        #print (len(TxWidth_0),len(IPPs_start_0))
        ##Determine if it's ULP
        if (len(TxWidth_0) != 100 or len(IPPs_start_0) != 100):
            numFile += 1                
        
        elif (all(IPPs_start_0[1:]-IPPs_start_0[:-1]==250000) and all(TxWidth_0==12500)):
            
            ##Determine if it's starting ULP

            numFile_ref = numFile + 18
            
            fph5=bz2.BZ2File(os.path.join(filespath,flist[numFile_ref]),'rb').read() 
            ndata_ref = read_file_data_bz(fph5)
            #ndata_ref = read_file_data(os.path.join(filespath,flist[numFile_ref]))
            TxWidth_ref, IPPs_start_ref = Detect_IPP_Tx(ndata_ref,False)
            if (len(TxWidth_ref) != 100 or len(IPPs_start_ref) != 100):
                numFile += 19   
            
            elif (all(IPPs_start_ref[1:]-IPPs_start_ref[:-1]==250000) and all(TxWidth_ref==12500)):
                psf_data = ndatai[IPPs_start_0[0]:IPPs_start_0[0] + TxWidth_0[0]]
                psf = psf_func(psf_data)

                spectrum_10s = pulse_process(numFile,file_length)

                ##Deconvolution by using clean algorithm
                spec_dirty1 = spectrum_10s[:4096,3184:7280]
                spec_dirty2 = spectrum_10s[:4096,9104:13200]

                spec_dirty1_norm = spec_dirty1/np.amax(spec_dirty1)
                psf_norm = psf/np.amax(psf)
                spec_dirty2_norm = spec_dirty2/np.amax(spec_dirty2)

                cleaned_spec = hogbom(spec_dirty1_norm,psf_norm,0.05)
                cleaned_spec2 = hogbom(spec_dirty2_norm,psf_norm,0.05)

                height,downshift_frequency,upshift_frequency,sp_downshift_frequency,sp_upshift_frequency=spline_fit(cleaned_spec,cleaned_spec2)

                s1 = time.strftime("%x,%X",time.gmtime(int(flist[numFile][3:-11])))[:2]
                s2 = time.strftime("%x,%X",time.gmtime(int(flist[numFile][3:-11])))[3:5]
                s3 = time.strftime("%x,%X",time.gmtime(int(flist[numFile][3:-11])))[6:8]
                s4 = time.strftime("%x,%X",time.gmtime(int(flist[numFile][3:-11])))[9:11]
                s5 = time.strftime("%x,%X",time.gmtime(int(flist[numFile][3:-11])))[12:14]
                s6 = time.strftime("%x,%X",time.gmtime(int(flist[numFile][3:-11])))[15:18]


                sstime = []
                sstime = [s1]+[s2]+[s3]+[s4]+[s5]+[s6]


                outfname = time.strftime("%X",time.gmtime(int(flist[numFile][3:-11])))
                np.savez_compressed(specfolder+outfname,
                    height = height,
                    sp_upshift_frequency =sp_upshift_frequency,
                    downshift_frequency = downshift_frequency,
                    upshift_frequency = upshift_frequency,
                    sp_downshift_frequency = sp_downshift_frequency,
                    time = sstime
                    )     

                numFile += 19
            
            else:
                numFile += 19
        else:
            numFile += 1


    print ("done")

if __name__ == "__main__":
    #dpath = '/mnt/kudeki-0/usrp/gr/'
    dpath = '/mnt/kudeki-3/'
    dirlist = sorted(glob.glob1(dpath,'*'))
    syear_arr,smonth_arr,sday_arr,shh_arr,smm_arr,sss_arr = [],[],[],[],[],[]
    for i,fname in enumerate(dirlist):
        syear_arr += [fname[:4]]
        smonth_arr += [fname[5:7]]
        sday_arr += [fname[8:10]]
        shh_arr += [fname[11:13]]
        smm_arr += [fname[14:16]]
        sss_arr += [fname[17:19]]

    ##i = 18
    ## for i in range(21,39):
    for i in range(22,24):
      process_spec(syear_arr[i],smonth_arr[i],sday_arr[i],shh_arr[i],smm_arr[i],sss_arr[i])    



