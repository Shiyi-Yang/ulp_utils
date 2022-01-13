import tensorflow as tf
import numpy as np
import time
import numpy.linalg as la
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

data = np.load('/home/yang158/Training/train_data/spectra_170000.npz')
state_par = data['X']
fitted_par = data['Y']
X = np.copy(state_par[:state_par.shape[0]//10000*9000])
Y = np.copy(fitted_par[:state_par.shape[0]//10000*9000])

X = X.astype(np.float32)
Y = Y.astype(np.float32)

X[:,0] /= 1e3
X[:,1] /= 1e3
#X[:,2] = np.log10(X[:,2])
X[:,2] /= 1e11
#Y[:,1] = np.log10(Y[:,1])
#Y[:,3] = np.log10(Y[:,3])
#Y[:,4] = np.log10(Y[:,4])

X_train = X[:int(X.shape[0]*0.8)]
#Y_train = Y
X_test = X[int(X.shape[0]*0.8):]

Y_tmp = Y[:,:4]
Y_tmp[:,1] = np.log10(Y_tmp[:,1])
Y_tmp[:,-1] = np.log10(Y_tmp[:,-1])
Y_train = Y_tmp[:int(Y_tmp.shape[0]*0.8)]
Y_test = Y_tmp[int(Y_tmp.shape[0]*0.8):]


#Y_tmp = np.log10(Y[:,-2]).reshape(-1,1)
#Y_tmp = np.e**(Y_tmp)
#Y_train = Y_tmp[:int(Y_tmp.shape[0]*0.8)]
#Y_test = Y_tmp[int(Y_tmp.shape[0]*0.8):]
#Y_train = np.log10(Y[:,-2]).reshape(-1,1)
#Y_train = np.e**(Y_train)



model = Sequential()
model.add(Dense(2000,input_shape=(3,)))
model.add(LeakyReLU())
#model.add(Dense(4000))
#model.add(LeakyReLU())
#model.add(Dense(2000))
#model.add(LeakyReLU())
#model.add(Dense(2000))
#model.add(LeakyReLU())
#model.add(Dense(2000))
#model.add(LeakyReLU())
#model.add(Dense(2000))
#model.add(LeakyReLU())
model.add(Dense(1000))
model.add(LeakyReLU())
#model.add(Dense(1000))
#model.add(LeakyReLU())
#model.add(Dense(1000))
#model.add(LeakyReLU())
#model.add(Dense(2000))
#model.add(LeakyReLU())
#model.add(Dense(1000))
#model.add(LeakyReLU())
#model.add(Dense(2000))
#model.add(LeakyReLU())
#model.add(Dense(500))
#model.add(LeakyReLU())
model.add(Dense(500))
model.add(LeakyReLU())
model.add(Dense(200))
model.add(LeakyReLU())
model.add(Dense(100))
model.add(LeakyReLU())
model.add(Dense(4,activation='linear'))

opt = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)
model.compile(loss='mean_squared_error',optimizer= opt)
checkpoint = ModelCheckpoint(filepath = '/home/yang158/Training/Models/sess1/17e4_points/par4_l5_1e-4_1024_20000',monitor='val_loss',save_best_only=True,mode='min')
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=12000)
#model.fit(X_train,Y_train,epochs=30000,batch_size=1024,callbacks=[checkpoint])
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=20000,batch_size=1024,callbacks=[checkpoint,es])



#model = tf.keras.models.load_model('/home/yang158/Training/Models/sess1/4e4_points/m1_exp_15000_lr1e-4')
#config = model.get_config()
#new_model = Sequential.from_config(config)

#new_model.set_weights(model.get_weights())
#opt = tf.keras.optimizers.Adam(learning_rate=1e-6,beta_1=0.9)

#new_model.compile(loss='mean_squared_error',optimizer= opt)
#checkpoint = ModelCheckpoint(filepath = '/home/yang158/Training/Models/sess2/4e4_points/m1_exp_10000_lr1e-6',monitor='loss',save_best_only=True,mode='min')

#new_model.fit(X_train,Y_train,epochs=10000,batch_size=128,callbacks=[checkpoint])

