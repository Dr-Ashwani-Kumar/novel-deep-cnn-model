import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score
from mealpy.swarm_based.HHO import BaseHHO as HHO
from mealpy.swarm_based.SSO import BaseSSO as SSO
from mealpy.Prop import BaseHHO as prop

def fitness_function1(solution, model, X_test, y_test):
    wei_to_train = model.get_weights()
    wei_to_train_1 = wei_to_train[8]
    wei_to_train_new = solution.reshape(wei_to_train_1.shape[0], wei_to_train_1.shape[1])
    wei_to_train[8] = wei_to_train_new
    model.set_weights(wei_to_train)
    pred = model.predict(y_test)
    pred = np.argmax(pred, axis=1)
    X_test = np.argmax(X_test, axis=1)
    acc = accuracy_score(X_test, pred)
    return acc

def main_weight_updation_optimization_HHO(curr_wei, model, X_test, y_test):
    problem_dict1 = {
        "fit_func": fitness_function1,
        "lb": [curr_wei.min(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "ub": [curr_wei.max(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "minmax": "max",
        "log_to": None,
        "save_population": False,
        "Curr_Weight": curr_wei,
        "Model_trained_Partial": model,
        "test_loader": X_test,
        "tst_lab": y_test,
    }
    model = HHO(problem_dict1, epoch=1, pop_size=10)
    best_position2, best_fitness2 = model.solve()
    # Glob_best_fit_2=model.history.list_global_best_fit
    return best_position2


def main_weight_updation_optimization_SSO(curr_wei, model, X_test, y_test):
    problem_dict1 = {
        "fit_func": fitness_function1,
        "lb": [curr_wei.min(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "ub": [curr_wei.max(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "minmax": "max",
        "log_to": None,
        "save_population": False,
        "Curr_Weight": curr_wei,
        "Model_trained_Partial": model,
        "test_loader": X_test,
        "tst_lab": y_test,
    }
    model = SSO(problem_dict1, epoch=1, pop_size=10)
    best_position2, best_fitness2 = model.solve()
    # Glob_best_fit_2=model.history.list_global_best_fit
    return best_position2

def main_weight_updation_optimization_prop(curr_wei, model, X_test, y_test):
    problem_dict1 = {
        "fit_func": fitness_function1,
        "lb": [curr_wei.min(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "ub": [curr_wei.max(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "minmax": "max",
        "log_to": None,
        "save_population": False,
        "Curr_Weight": curr_wei,
        "Model_trained_Partial": model,
        "test_loader": X_test,
        "tst_lab": y_test,
    }
    model = prop(problem_dict1, epoch=1, pop_size=10)
    best_position2, best_fitness2 = model.solve()
    # Glob_best_fit_2=model.history.list_global_best_fit
    return best_position2


def main_update_hyperparameters_HHO(model, X_test, y_test):
    wei_to_train = model.get_weights()
    wei_to_train_11 = wei_to_train[8]
    wei_to_train_new = main_weight_updation_optimization_HHO(wei_to_train_11, model, X_test, y_test)
    wei_to_train_new = wei_to_train_new.reshape(wei_to_train_11.shape[0], wei_to_train_11.shape[1])
    wei_to_train[8] = wei_to_train_new
    model.set_weights(wei_to_train)
    return model


def main_update_hyperparameters_SSO(model, X_test, y_test):
    wei_to_train = model.get_weights()
    wei_to_train_11 = wei_to_train[8]
    wei_to_train_1 = main_weight_updation_optimization_SSO(wei_to_train_11, model, X_test, y_test)
    wei_to_train_1_new = wei_to_train_1.reshape(wei_to_train_11.shape[0], wei_to_train_11.shape[1])
    wei_to_train[8] = wei_to_train_1_new
    model.set_weights(wei_to_train)
    return model

def main_update_hyperparameters_PROP(model, X_test, y_test):
    wei_to_train = model.get_weights()
    wei_to_train_11 = wei_to_train[8]
    wei_to_train_1 = main_weight_updation_optimization_prop(wei_to_train_11, model, X_test, y_test)
    wei_to_train_1_new = wei_to_train_1.reshape(wei_to_train_11.shape[0], wei_to_train_11.shape[1])
    wei_to_train[8] = wei_to_train_1_new
    model.set_weights(wei_to_train)
    return model


def DeepCNN(X_train, y_train, X_test, y_test, epochs,opt):
    ## Sequential Layer
    model = Sequential()
    ## First Convolutional Layer
    model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1))),
    ## Maxpooling Layer
    model.add(MaxPooling2D(1, 1)),
    ## Second Convolutional Layer
    model.add(Conv2D(32, 3, padding='same', activation='relu')),
    ## Maxpooling Layer
    model.add(MaxPooling2D(1, 1)),
    ## Third Convolutional Layer
    model.add(Conv2D(64, 3, padding='same', activation='relu')),
    ## Maxpooling Layer
    model.add(MaxPooling2D(1,1)),
    ## Dropout Layer
    model.add(Dropout(0.2)),
    ## Flatten Layer
    model.add(Flatten()),
    ## Fully Connected Layer
    model.add(Dense(64, activation='relu')),
    model.add(Dense(y_test.shape[1],  activation='softmax')),
    model.compile(optimizer='adam', loss="mse", metrics=['accuracy']),
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=32),
    if opt == 0:
        opt_model = model
    elif opt == 1:
        ## HHO Optimization Algorithm
        opt_model = main_update_hyperparameters_HHO(model, X_test, y_test)
    elif opt == 2:
        ## SSO Optimization Algorithm
        opt_model = main_update_hyperparameters_SSO(model, X_test, y_test)
    else:
        ## Proposed Optimization Algorithm
        opt_model = main_update_hyperparameters_PROP(model, X_test, y_test)
    preds = opt_model.predict(X_test)
    pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return pred, y_true
