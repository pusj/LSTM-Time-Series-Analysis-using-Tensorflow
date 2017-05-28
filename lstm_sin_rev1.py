import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import warnings

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from sklearn.metrics import mean_squared_error

from lstm_predictor import generate_data, lstm_model

warnings.filterwarnings("ignore")

LOG_DIR = 'resources/logs/'
TIMESTEPS = 1
RNN_LAYERS = [{'num_units': 400}]
DENSE_LAYERS = None
TRAINING_STEPS = 500
PRINT_STEPS = TRAINING_STEPS # / 10
BATCH_SIZE = 100

regressor = SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),))
                          #   model_dir=LOG_DIR)

X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)


noise_train = np.asmatrix(np.random.normal(0,0.2,len(y['train'])),dtype = np.float32)
noise_val = np.asmatrix(np.random.normal(0,0.2,len(y['val'])),dtype = np.float32)
noise_test = np.asmatrix(np.random.normal(0,0.2,len(y['test'])),dtype = np.float32) #asmatrix

noise_train = np.transpose(noise_train)
noise_val = np.transpose(noise_val)
noise_test = np.transpose(noise_test)


y['train'] = np.add( y['train'],noise_train)
y['val'] = np.add( y['val'],noise_val)
y['test'] = np.add( y['test'],noise_test)


# print(type(y['train']))


print('-----------------------------------------')
print('train y shape',y['train'].shape)
print('train y shape_num',y['train'][1:5])
print('noise_train shape',noise_train.shape)
print('noise_train shape_num',noise_train.shape[1:5])


# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],)
                                                     # every_n_steps=PRINT_STEPS,)
                                                     # early_stopping_rounds=1000)
# print(X['train'])
# print(y['train'])

SKCompat(regressor.fit(X['train'], y['train'],
              monitors=[validation_monitor],
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS))

print('X train shape', X['train'].shape)
print('y train shape', y['train'].shape)

print('X test shape', X['test'].shape)
print('y test shape', y['test'].shape)
predicted = np.asmatrix(regressor.predict(X['test']),dtype = np.float32) #,as_iterable=False))
predicted = np.transpose(predicted)


rmse = np.sqrt((np.asarray((np.subtract(predicted, y['test']))) ** 2).mean()) 
# this previous code for rmse was incorrect, array and not matricies was needed: rmse = np.sqrt(((predicted - y['test']) ** 2).mean())  
score = mean_squared_error(predicted, y['test'])
nmse = score / np.var(y['test']) # should be variance of original data and not data from fitted model, worth to double check

print("RSME: %f" % rmse)
print("NSME: %f" % nmse)
print("MSE: %f" % score)



plot_test, = plt.plot(y['test'], label='test')
plot_predicted, = plt.plot(predicted, label='predicted')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
