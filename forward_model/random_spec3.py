import numpy as np
import sys,os
import time
import random
from gen_model import forward_model_gpu_T
import numpy.linalg as la
from scipy import optimize 
from scipy.optimize import least_squares as lsq

def gen_spec(freq,coeff,par):
    return par[0]*np.exp(-(abs(freq-coeff)/par[1])**par[2])\
            + (1-par[0])*np.exp(-(1./2)*(abs(freq-coeff)/par[3])**2)

def get_residual(par, freq,mu, y_spec):
    ##par[0]: amplitude of ggauss
    ##par[1]: width of ggauss
    ##par[2]: power of gauss
    ##par[3]: width of gauss
    
    return par[0]*np.exp(-(abs(freq-mu)/par[1])**par[2])\
            + (1-par[0])*np.exp(-(1./2)*(abs(freq-mu)/par[3])**2) - y_spec

T = 1e-5
ion_composition = 0.9
data = np.load('/home/yang158/notebook/ISR_model/single_height/Table/Data/random_spectra19.npz')
Te_arr = data['Te_arr']
Ti_arr = data['Ti_arr']
Ne_arr = data['Ne_arr']



N = 1e7
N_s = 1e7
N_c = 5e7
f2 = np.linspace(1e6,11e6,int(N_s),endpoint = False)
f_norm = np.linspace(1,11,int(N_s),endpoint = False)
df1 = 1e5/N_c
df2 = 1
ratio = int(df2/df1)
#gpu_arr = np.zeros((1250,10),np.float64)
gpu_arr = np.zeros((2500,10),np.float64)
Tefolder = '/home/yang158/random_spec_table19/'
if not os.path.exists(Tefolder):
    os.makedirs(Tefolder)

par_arr,mu_arr,amp_arr = [],[],[]
t1 = time.time()
for i in range(gpu_arr.shape[0]):
    print(i)
    #spec_temp = forward_model_gpu_T.gen_forward_model(f2,T,Ne_arr[i+2*gpu_arr.shape[0]],Te_arr[i+2*gpu_arr.shape[0]],Ti_arr[i+2*gpu_arr.shape[0]],ion_composition)
    spec_gpu = forward_model_gpu_T.gen_forward_model(f2,T,Ne_arr[i+2*gpu_arr.shape[0]],Te_arr[i+2*gpu_arr.shape[0]],Ti_arr[i+2*gpu_arr.shape[0]],ion_composition)
    #pl = fo[np.argmax(spec_temp)]
    #pl = f2[np.argmax(spec_temp)]
    pl = f2[np.argmax(spec_gpu)]

    #print("pl obtained and fp=%f"%(fp))

    f_start = int(pl)-0.5e5
    f_end = int(pl) + 0.5e5
    f1 = np.linspace(f_start,f_end,int(N_c),endpoint = False)
    spec_c = forward_model_gpu_T.gen_forward_model(f1,T,Ne_arr[i+2*gpu_arr.shape[0]],Te_arr[i+2*gpu_arr.shape[0]],Ti_arr[i+2*gpu_arr.shape[0]],ion_composition)
    #spec_gpu = forward_model_gpu_T.gen_forward_model(f2,T,Ne[-20],Te[i],Ti,ion_composition)
    #spec_gpu = np.copy(spec_temp)
    spec_c = spec_c.reshape(len(spec_c)//ratio,ratio).max(axis=-1)

    for freq in range(len(f2)):
        if f2[freq] == f_start:
            print("The frequency is %f and f_start is %f,"%(f2[freq]/1e6,f_start/1e6))
            spec_gpu[freq:freq + len(spec_c)] = spec_c
            break

    #spec_gpu = spec_gpu.reshape(len(spec_gpu)//2000,2000).max(axis=-1)
    
    index = np.argmax(spec_gpu)
    mu = f_norm[index]
    sub_f = f_norm[index-int(1e5):index+int(1e5)]
    ref = spec_gpu[index-int(1e5):index+int(1e5)]
    ref = ref/ref.max()
    fwhm = (sub_f[ref>=ref.max()/2])[-1]-sub_f[ref>=ref.max()/2][0]
    sig = fwhm/2/np.sqrt(2*np.log(2))
    coeff = mu
    par = [0.5,sig,2,sig]
    res = lsq(get_residual,par,args=(sub_f,coeff,ref),
              bounds = ([0,0,0,0],[1,np.inf,5,np.inf]))
    par_arr +=[res.x]
    amp_arr += [spec_gpu.max()]
    mu_arr += [mu]  
    
    #gpu_arr[i] = spec_gpu[index-int(1e5):index+int(1e5)]  
    
t2 = time.time()
#np.savez(Tefolder+'spec3', gpu_arr = gpu_arr,par_arr = par_arr,amp_arr = amp_arr, mu_arr = mu_arr, time=t2-t1)     
np.savez(Tefolder+'spec3',par_arr = par_arr,amp_arr = amp_arr, mu_arr = mu_arr, time=t2-t1)     


        
    
