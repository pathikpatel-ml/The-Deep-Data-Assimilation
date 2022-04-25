
#from AnDA_stat_functions import normalise
import numpy as np
import tensorflow as tf
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4))
from AnDA_generate_data import AnDA_generate_data
#import openvino_tensorflow
#openvino_tensorflow.set_backend('CPU')
#from data import AnDA_generate_data
# convert an array of values into a dataset matrix
'''
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
'''
def model_creation():
    from tensorflow import keras
    #from AnDA_generate_data import AnDA_generate_data
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    # from AnDA_dynamical_models import AnDA_Lorenz_63, AnDA_Lorenz_96
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    # from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import LSTM
    # import preprocessing
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pickle
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    # from deployml.keras import NeuralNetworkBase
    from keras.layers import Bidirectional
    from keras.layers import Dropout
    from keras import Model
    from keras.layers import Input, Dense, Bidirectional
    from keras.layers.recurrent import LSTM
    from keras.models import model_from_json
    import os
    from sklearn.model_selection import RepeatedKFold
    from numpy import mean
    from numpy import std
    from matplotlib import pyplot
    class GD:
        model = 'Lorenz_63'

        class parameters:
            sigma = 10.0
            rho = 28.0
            beta = 8.0 / 3

        dt_integration = 0.01  # integration time
        dt_states = 1  # number of integeration times between consecutive states (for xt and catalog)
        dt_obs = 8  # number of integration times between consecutive observations (for yo)
        var_obs = np.array([0])  # indices of the observed variables
        nb_loop_train = 10 ** 4  # size of the catalog
        nb_loop_test = 10  # size of the true state and noisy observations
        sigma2_catalog = 0.0  # variance of the model error to generate the catalog
        sigma2_obs = 2.0  # variance of the observation error to generate observation

    '''    
    def AnDA_Lorenz_63(S, t, sigma, rho, beta):
        """ Lorenz-63 dynamical model. """
        x_1 = sigma * (S[1] - S[0]);
        x_2 = S[0] * (rho - S[2]) - S[1];
        x_3 = S[0] * S[1] - beta * S[2];
        dS = np.array([x_1, x_2, x_3]);
        return dS

    class GD:
        model = 'Lorenz_63'

        class parameters:
            sigma = 10.0
            rho = 28.0
            beta = 8.0 / 3

        dt_integration = 0.01  # integration time
        dt_states = 1  # number of integeration times between consecutive states (for xt and catalog)
        dt_obs = 8  # number of integration times between consecutive observations (for yo)
        var_obs = np.array([0])  # indices of the observed variables
        nb_loop_train = 10 ** 2  # size of the catalog
        nb_loop_test = 10  # size of the true state and noisy observations
        sigma2_catalog = 0.0  # variance of the model error to generate the catalog
        sigma2_obs = 2.0  # variance of the observation error to generate observation

    def AnDA_generate_data(GD):
        """ Generate the true state, noisy observations and catalog of numerical simulations. """
￼
        # initialization
        class xt:
            values = [];
            time = [];

        class yo:
            values = [];
            time = [];

        class catalog:
            analogs = [];
            successors = [];
            source = [];

        # test on parameters
        if GD.dt_states > GD.dt_obs:
            print('Error: GD.dt_obs must be bigger than GD.dt_states');
        if (np.mod(GD.dt_obs, GD.dt_states) != 0):
            print('Error: GD.dt_obs must be a multiple of GD.dt_states');

        # use this to generate the same data for different simulations
        np.random.seed(1);

        if (GD.model == 'Lorenz_63'):
            # 5 time steps (to be in the attractor space)
            # x0 = np.array([8.0, 0.0, 30.0]);
            x0 = np.random.rand(3)
            S = odeint(AnDA_Lorenz_63, x0, np.arange(0, 5 + 0.000001, GD.dt_integration),
                       args=(GD.parameters.sigma, GD.parameters.rho, GD.parameters.beta));
            x0 = S[S.shape[0] - 1, :];

            # generate true state (xt)
            S = odeint(AnDA_Lorenz_63, x0, np.arange(0.01, GD.nb_loop_test + 0.000001, GD.dt_integration),
                       args=(GD.parameters.sigma, GD.parameters.rho, GD.parameters.beta));
            T_test = S.shape[0];
            t_xt = np.arange(0, T_test, GD.dt_states);
            xt.time = t_xt * GD.dt_integration;
            xt.values = S[t_xt, :];

            # generate  partial/noisy observations (yo)
            eps = np.random.multivariate_normal(np.zeros(3), GD.sigma2_obs * np.eye(3, 3), T_test);
            yo_tmp = S[t_xt, :] + eps[t_xt, :];
            t_yo = np.arange(0, T_test, GD.dt_obs);
            i_t_obs = np.where((np.in1d(t_xt, t_yo)) == True)[0];
            yo.values = xt.values * np.nan;
            yo.values[np.ix_(i_t_obs, GD.var_obs)] = yo_tmp[np.ix_(i_t_obs, GD.var_obs)];
            yo.time = xt.time;

            # generate catalog
            S = odeint(AnDA_Lorenz_63, S[S.shape[0] - 1, :],
                       np.arange(0.01, GD.nb_loop_train + 0.000001, GD.dt_integration),
                       args=(GD.parameters.sigma, GD.parameters.rho, GD.parameters.beta));
            T_train = S.shape[0];
            eta = np.random.multivariate_normal(np.zeros(3), GD.sigma2_catalog * np.eye(3, 3), T_train);
            catalog_tmp = S + eta;
            catalog.analogs = catalog_tmp[0:-GD.dt_states, :];
            catalog.successors = catalog_tmp[GD.dt_states:, :];
            catalog.source = GD.parameters;

        #np.random.seed()

        return catalog, xt, yo;
    '''
    catalog, xt, yo = AnDA_generate_data(GD)
    '''
    print()
    print("catalog.analogs")
    print(catalog.analogs)
    print("catalog.successor")
    print(catalog.successors)
    '''
    '''
    line1,=plt.plot(xt.time,xt.values[:,0],'-r');plt.plot(yo.time,yo.values[:,0],'.r')
    line2,=plt.plot(xt.time,xt.values[:,1],'-b');plt.plot(yo.time,yo.values[:,1],'.b')
    line3,=plt.plot(xt.time,xt.values[:,2],'-g');plt.plot(yo.time,yo.values[:,2],'.g')
    plt.xlabel('Lorenz-63 times')49741
    plt.legend([line1, line2, line3], ['$x_1$', '$x_2$', '$x_3$'])
    plt.title('Lorenz-63 true (continuous lines) and observed trajectories (points)')
    plt.show()
    # catalog sample of simulated trajectories
    print('######################')
    print('SAMPLE OF THE CATALOG:')
    print('######################')
    print('Analog (t1): ' + str(catalog.analogs[0,:]), end=" ")
    print('Successor (t1+dt): ' + str(catalog.successors[0,:]))
    print('Analog (t2): ' + str(catalog.analogs[10,:]), end=" ")
    print('Successor (t2+dt): ' + str(catalog.successors[10,:]))
    print('Analog (t3): ' + str(catalog.analogs[20,:]), end=" ")
    print('Successor (t3+dt): ' + str(catalog.successors[20,:]))
    '''
    
    X = catalog.analogs
    y = catalog.successors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_test.shape)
    print("")
    print(test_X.shape)
    print(train_X.shape, y_train.shape, test_X.shape, y_test.shape)
    model = Sequential()
    model.add(LSTM(20,activation='relu',input_shape=(train_X.shape[1], train_X.shape[2])))
    #model.add(Dropout(0.1))
    model.add(Dense(8))
    #model.add(Dense(5))
    model.add(Dense(3))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, y_train, epochs=15, batch_size= 50, validation_data=(test_X, y_test), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # get the model
    '''
    def get_model(n_inputs, n_outputs):
        model = Sequential()
        model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        #Smodel.add(Dense(8))
        #model.add(Dense(4))
        model.add(Dense(n_outputs))
        model.compile(loss='mae', optimizer='adam')
        return model

    
    print(X_test)
    X_train = X_train.reshape(-1, 1, 3)
    X_test  = X_test.reshape(-1, 1, 3)
    print(X_test)
    y_train = y_train.reshape(-1, 1, 3)
    y_test = y_test.reshape(-1, 1, 3)
    #print(X_train)

    n_past = 10
    n_future = 10 
    n_features = 3

    model = keras.models.Sequential()
    model.add(LSTM(36,activation='softplus' , kernel_initializer='lecun_uniform',input_shape=(10,3), return_sequences=True))
    model.add(LSTM(12,return_sequences=True))
    #model.add(LSTM(12, activation='relu',return_sequences=True))
    ##model.add(LSTM(3, kernel_initializer='lecun_uniform', return_sequences=True))
    model.add(Dropout(0.01))
    model.add(LSTM(3, return_sequences=True,activation='softplus'))
    opt = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.01, nesterov=False, name='adam' )
    model.compile(loss="mean_squared_error",optimizer=opt, metrics= ['accuracy'])

    model.fit(X_train,y_train, batch_size=100,epochs=350, validation_data=(X_test,y_test))
    model.summary()
    #model = Sequential()
    #model.add(LSTM(3, activation='relu',input_shape=(3 , 3) )
    #model.add(Dense(3))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #model.fit(X_train, y_train, epochs=350, batch_size=100, verbose=2)
    '''
    
    #x1 = X_test[1:11, :]
    #print(y_test[1:11, :])
    #ans = model.predict(x1)
    #print(ans)
    # !mkdir -p saved_model
    #model.save('saved_model_new_2/my_model11')
    # print("Saved model to disk")
    # filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    model_json = model.to_json()
    with open("model_new_1.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model_1.h5")
    #print("Saved model to disk")
    # joblib.dump(model, "model.pkl")
    # with open('model.pkl', 'wb') as f: netcdf
    # pickle.dump(model, f)
    # with open('models/basic_history.pickle', 'wb') as f:
    #   pickle.dump(history.history, f)
    #from numpy import concatenate
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled = scaler.fit_transform(X,y)
    from math import sqrt
    print("Saved model to disk")
    """" Apply the analog method on catalog of historical data to generate forecasts. """
    yhat = model.predict(test_X)
    print(test_X.shape)
    print(yhat.shape)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, yhat))
    print('Test RMSE: %.3f' % rmse)
    
    
    

model_creation()

