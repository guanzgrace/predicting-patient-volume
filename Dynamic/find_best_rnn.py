import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import time

# This was run on AWS because the average CPU time per model was 20 min.
# We save the trained model weights to h5 files so they can be reloaded for
# confirmation of best model (via validation MSE) and for base use in OMA and
# OSA models.


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

############# Copying MLHCExperiment-Common-Val17Test18.ipynb; not all of this is used. #############

TRAIN_START = pd.datetime(2010, 1, 1)
TRAIN_END = pd.datetime(2016, 12, 31)
VAL_START = pd.datetime(2017, 1, 1)
VAL_END = pd.datetime(2017, 12, 31)
TEST_START = pd.datetime(2018, 1, 1)
TEST_END = pd.datetime(2018, 9, 20)

BASE_FEATURES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                 '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                 '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                 '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
                 '50', '51', '52', 'isSunday', 'isMonday', 'isTuesday', 'isWednesday', 
                 'isThursday', 'isFriday', 'isHoliday', 'TEMP_normalized',
                 'WIND_normalized', 'SNOW_normalized', 'PRCP_normalized']

SEVEN_DAY_LAG = ['Shift-1', 'Shift-2', 'Shift-3', 'Shift-4', 'Shift-5', 'Shift-6', 'Shift-7']

SEVEN_DAY_LAG_ICD8 = ['Shift-1ICD8', 'Shift-2ICD8', 'Shift-3ICD8', 'Shift-4ICD8',
                      'Shift-5ICD8', 'Shift-6ICD8', 'Shift-7ICD8']

SEVEN_DAY_LAG_ICD8OPP = ['Shift-1ICD8opp', 'Shift-2ICD8opp', 'Shift-3ICD8opp',
                         'Shift-4ICD8opp', 'Shift-5ICD8opp', 'Shift-6ICD8opp',
                         'Shift-7ICD8opp']

ONE_DAY_LAG = ['Shift-1']

ONE_DAY_LAG_ICD8 = ['Shift-1ICD8']

ONE_DAY_LAG_ICD8OPP = ['Shift-1ICD8opp']

SEVEN_DAY_LAG_FEATURES = BASE_FEATURES + SEVEN_DAY_LAG
SEVEN_DAY_LAG_ICD8_FEATURES = BASE_FEATURES + SEVEN_DAY_LAG_ICD8
SEVEN_DAY_LAG_ICD8OPP_FEATURES = BASE_FEATURES + SEVEN_DAY_LAG_ICD8OPP

ONE_DAY_LAG_FEATURES = BASE_FEATURES + ONE_DAY_LAG
ONE_DAY_LAG_ICD8_FEATURES = BASE_FEATURES + ONE_DAY_LAG_ICD8
ONE_DAY_LAG_ICD8OPP_FEATURES = BASE_FEATURES + ONE_DAY_LAG_ICD8OPP

ALL_VISITS_LABEL = "AdjCount"
ICD8_LABEL = "ICD8perFTE"
ICD8OPP_LABEL = "ICD8oppperFTE"

m_train_end = [pd.datetime(2017, 12, 31), pd.datetime(2018, 1, 31), pd.datetime(2018, 2, 28),
                       pd.datetime(2018, 3, 31), pd.datetime(2018, 4, 30), pd.datetime(2018, 5, 31),
                       pd.datetime(2018, 6, 30), pd.datetime(2018, 7, 31), pd.datetime(2018, 8, 31)]
m_test_start = [pd.datetime(2018, 1, 1), pd.datetime(2018, 2, 1), pd.datetime(2018, 3, 1),
                      pd.datetime(2018, 4, 1), pd.datetime(2018, 5, 1), pd.datetime(2018, 6, 1),
                      pd.datetime(2018, 7, 1), pd.datetime(2018, 8, 1), pd.datetime(2018, 9, 1)]
m_test_end = [pd.datetime(2018, 1, 31), pd.datetime(2018, 2, 28), pd.datetime(2018, 3, 31),
                     pd.datetime(2018, 4, 30), pd.datetime(2018, 5, 31), pd.datetime(2018, 6, 30),
                     pd.datetime(2018, 7, 31), pd.datetime(2018, 8, 31), pd.datetime(2018, 9, 20)]


'''
Reads our dataset into a pandas dataframe and converts the dataframe
index into datetime objects.

'''

def read_data():
    # this needs to go up one directory since all files using it are inside a subdirectory
    visits = pd.read_csv("../DatasetForMLHC.csv") 
    visits.index = pd.to_datetime(visits["Unnamed: 0"])
    visits.index = pd.DatetimeIndex(visits.index.values)
    visits = visits.reindex(visits.asfreq('d').index, fill_value=0)
    return visits


'''
HELPER METHODS FOR NN AND RNN METHODS
'''

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


'''
Spilts the dataset into training, validation, and testing, for our dynamic,
OSH and OMA baseline, NN, and RNN models.
'''
def split_for_baseline_and_nn(X, y):
    X_train = X.loc[TRAIN_START:TRAIN_END]
    y_train = y.loc[TRAIN_START:TRAIN_END]
    X_val = X.loc[VAL_START:VAL_END]
    y_val = y.loc[VAL_START:VAL_END]
    X_test = X.loc[TEST_START:TEST_END]
    y_test = y.loc[TEST_START:TEST_END]
    return X_train, y_train, X_val, y_val, X_test, y_test

'''
Builds and compiles our NN model.
'''
def build_nn_model(X_train):
    model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model

'''
Get X_train_t and X_test_t into the correct shape for the RNN model. For example:
X_train_t = np.array(X_train_scaled).reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_t = np.array(X_test_scaled).reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
'''
def reshape_for_rnn(a):
    return np.array(a).reshape(a.shape[0], 1, a.shape[1])

'''
Builds and compiles our RNN model.
'''
def build_rnn_model(X_train):
    model = keras.Sequential()
    model.add(layers.LSTM(64, activation=tf.nn.relu, input_shape=(1, X_train.shape[1]),
                kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model

'''
Builds and compiles our RNN model.
'''
def build_rnn_model1(X_train):
    model = keras.Sequential()
    model.add(layers.LSTM(64, activation=tf.nn.relu, input_shape=(1, X_train.shape[1]),
                kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model


'''
Builds and compiles our RNN model.
'''
def build_rnn_model2(X_train):
    model = keras.Sequential()
    model.add(layers.LSTM(64, activation=tf.nn.relu, input_shape=(1, X_train.shape[1]),
                kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(layers.LSTM(16, return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model

'''
Builds and compiles our RNN model.
'''
def build_rnn_model3(X_train):
    model = keras.Sequential()
    model.add(layers.LSTM(32, activation=tf.nn.relu, input_shape=(1, X_train.shape[1]),
                kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model

'''
Builds and compiles our RNN model.
'''
def build_rnn_model4(X_train):
    model = keras.Sequential()
    model.add(layers.LSTM(32, activation=tf.nn.relu, input_shape=(1, X_train.shape[1]),
                kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(layers.LSTM(16, return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model

'''
Builds and compiles our RNN model.
'''
def build_rnn_model5(X_train):
    model = keras.Sequential()
    model.add(layers.LSTM(32, activation=tf.nn.relu, input_shape=(1, X_train.shape[1]),
                kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(layers.LSTM(8, return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    #model.summary()
    return model

'''
Prints the mean and standard deviation for an array of MSEs.
'''
def print_mse_metrics(mses):
    print("MSE and RMSE over 10 trials with standard deviation in parentheses")
    print("Average MSE: %.3f (%.3f)\nAverage RMSE: %.3f (%.3f)"
          % (np.mean(mses), np.std(mses), np.mean(np.sqrt(mses)), np.std(np.sqrt(mses))))

'''
Gets the monthly split, next, and end dates for a given index.
'''
def get_split_dates(i):
    if i > 9:
        return -1, -1, -1
    return m_train_end[i], m_test_start[i], m_test_end[i]

'''
Gets the dynamic split and next dates.
'''
def get_dynamic_split_dates():
    return m_train_end[0], m_test_start[0]



def run_dynamic_rnn(model, name, features, label, epochs, patience, trials):
    print("\n\n")
    print(name)
    X = all_visits[features]
    y = all_visits[label]

    mses = list()
    val_mses = list()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',
                                               patience=patience, restore_best_weights=True)

    X_train, y_train, X_val, y_val, X_test, y_test = split_for_baseline_and_nn(X, y)
    for j in range(trials):
        start = time.time()
        print("Trial %d" % (j + 1))

        features_min_max = preprocessing.MinMaxScaler()

        mod = model(X_train)

        X_train_t = reshape_for_rnn(features_min_max.fit_transform(X_train))
        X_val_t = reshape_for_rnn(features_min_max.transform(X_val))

        history = mod.fit(X_train_t, y_train.values, epochs=epochs,
                  batch_size=1, shuffle=False, 
                  validation_data=[X_val_t, y_val.values],
                  verbose=0, callbacks=[PrintDot(), early_stop])

        val_mses.append(min(history.history["val_mean_squared_error"]))

        # serialize weights to HDF5
        model_name = name + "-Trial" + str(j) + "-" + str(time.time()) + ".h5"
        mod.save_weights(model_name)

        X_test_t = reshape_for_rnn(features_min_max.transform(X_test))
            
        y_pred = mod.predict(X_test_t, batch_size=1)
        mses.append(mean_squared_error(y_pred, y_test))
        end = time.time()
        print("Trial completed in %.2f s" % (end - start))
        print("Average MSE so far: %.3f (%.3f)" % (np.mean(mses), np.std(mses)))
        print("Average Validation MSE so far: %.3f (%.3f)" % (np.mean(val_mses), np.std(val_mses)))
    
    print("MSEs")
    print_mse_metrics(mses)
    print("Validation MSEs")
    print_mse_metrics(val_mses)
    return val_mses, mses, y_pred


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################


all_visits = read_data()
all_visits = all_visits.where(all_visits["AdjCount"] > 0).dropna()

epochs = 300
patience = 5
trials = 5

models = dict()
models["64-64"] = build_rnn_model # 64-64
models["64-32"] = build_rnn_model1 # 64-32
models["64-16"] = build_rnn_model2 # 64-16
models["32-32"] = build_rnn_model3 # 32-32
models["32-16"] = build_rnn_model4 # 32-16
models["32-8"] = build_rnn_model5 # 32-8


best_model = "64-64"
min_mse = 9999
for key, value in models.items():
    print(key)
    val_mses, mses, _ = run_dynamic_rnn(value, "Dynamic-RNN-AllVisits-" + str(key),
                   BASE_FEATURES, ALL_VISITS_LABEL, epochs, patience, 5)
    print(val_mses)
    if np.mean(val_mses) < min_mse:
        min_mse = np.mean(val_mses)
        best_model = key
    print("Best model %s" % (best_model))
    print("Min Val MSE %.3f" % (min_mse))

