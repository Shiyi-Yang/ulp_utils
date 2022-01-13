import random, sys, os, glob
import numpy as np
dirpath = '/home/yang158/notebook/ISR_model/single_height/Table/Data/'
flist = sorted(glob.glob1(dirpath,'random_spec*'))
Te,Ti,Ne = [],[],[]
for i in range(len(flist)):
    data = np.load(dirpath + flist[i])
    Te_arr = data['Te_arr']
    Ti_arr = data['Ti_arr']
    Ne_arr = data['Ne_arr']
    Te = Te + list(Te_arr)
    Ti = Ti + list(Ti_arr)
    Ne = Ne + list(Ne_arr)
Te = np.array(Te)
Ti = np.array(Ti)
Ne = np.array(Ne)
sample_set = {(Te[i],Ti[i],Ne[i]) for i in range(len(Te))}
duplicate = set()
assert(len(sample_set)==len(flist)*10000)

Te_arr = []
Ti_arr = []
Ne_arr = []
Te_not = []
Ti_not = []
Ne_not = []

while (len(Te_arr)<10000):
    Ti = random.uniform(500,2500)
    Te = random.uniform(500,2500)
    if (Te < 0.9 * Ti):
        Te_not += [Te]
        Ti_not += [Ti]
        Ne_not += [10**random.uniform(10.8,12)]
    else:
        Ne_tmp = 10**random.uniform(10.8,12)
        if (Te,Ti,Ne_tmp) not in sample_set:
            Te_arr += [Te]
            Ti_arr += [Ti]
            Ne_arr += [Ne_tmp]

np.savez('/home/yang158/notebook/ISR_model/single_height/Table/Data/random_spectra6',
         Te_arr = Te_arr,Ti_arr = Ti_arr,Ne_arr = Ne_arr,Te_not = Te_not,Ti_not = Ti_not,
        Ne_not = Ne_not)
