%pylab inline
import glob
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

def load_state_par():
    dpath = '/home/yang158/notebook/ISR_model/single_height/Data/'
    flist = sorted(glob.glob1(dpath,'random_spec*'))
    Te,Ti,Ne = [],[],[]
    for i in range(len(flist)):
        data = np.load(dpath+flist[i])
        Te_arr = data['Te_arr']
        Ti_arr = data['Ti_arr']
        Ne_arr = data['Ne_arr']
        Te += list(Te_arr)
        Ti += list(Ti_arr)
        Ne += list(Ne_arr)
    Te = array(Te)
    Ti = array(Ti)
    Ne = array(Ne)
    state_par = (array(list(Te)+list(Ti)+list(Ne)).reshape(3,-1)).T ## Te,Ti,Ne
    return state_par

def load_Gaussian_par():
    dirpath = '/home/yang158/'
    dirlist = sorted(glob.glob1(dirpath,'random_spec*'))
    fitted_par = np.zeros((len(dirlist)*10000,6))
    for i in range(len(dirlist)):
        flist = sorted(glob.glob1(dirpath+dirlist[i],'*'))
        #print(flist)
        for j in range(len(flist)):
            data = load(dirpath+dirlist[i]+'/'+flist[j])
            print(dirlist[i] +'/' + flist[j])
            #spec = data['gpu_arr']
            par_arr = data['par_arr']
            amp_arr = data['amp_arr']
            mu_arr = data['mu_arr']
            #spec_arr[i*spec.shape[0]:(i+1)*spec.shape[0]] = spec
            par_arr = concatenate((par_arr,amp_arr.reshape(-1,1)),axis=1)
            par_arr = concatenate((par_arr,mu_arr.reshape(-1,1)),axis=1)
            fitted_par[i*10000+j*par_arr.shape[0]:i*10000+(j+1)*par_arr.shape[0]] = par_arr
    return fitted_par

state_par = load_state_par()
fitted_par = load_Gaussian_par()

X = array([state_par[i] for i in range(state_par.shape[0])
           if fitted_par[i][-2] > 2e4])
Y = array([fitted_par[i] for i in range(fitted_par.shape[0])
           if fitted_par[i][-2] > 2e4])

np.savez('/home/yang158/notebook/ISR_model/single_height/Interpolation/spectra_170000',X=X,Y=Y)
