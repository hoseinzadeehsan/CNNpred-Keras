import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from os.path import join
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from pathlib2 import Path
from tensorflow.keras import backend as K, callbacks
import tensorflow as tf
import tensorflow.keras as keras



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2


def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date') # parse_dates=['Date'])
    except IOError:
        print("IO ERROR")
    return df_raw


def costruct_data_warehouse(ROOT_PATH, file_names):
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    global order_stocks
    data_warehouse = {}

    for stock_file_name in file_names:

        file_dir = os.path.join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        number_of_stocks += 1

        data = df_raw
        df_name = data['Name'][0]
        order_stocks.append(df_name)
        del data['Name']

        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        data = data[:-predict_day]
        target.index = data.index
        # Becasue of using 200 days Moving Average as one of the features
        data = data[200:]
        data = data.fillna(0)
        data['target'] = target
        target = data['target']
        # data['Date'] = data['Date'].apply(lambda x: x.weekday())
        del data['target']

        number_feature = data.shape[1]
        samples_in_each_stock = data.shape[0]

        train_data = data[data.index < '2016-04-21']
        train_data1 = scale(train_data)
        # print train_data.shape
        train_target1 = target[target.index < '2016-04-21']
        train_data = train_data1[:int(0.75 * train_data1.shape[0])]
        train_target = train_target1[:int(0.75 * train_target1.shape[0])]

        valid_data = scale(train_data1[int(0.75 * train_data1.shape[0]) - seq_len:])
        valid_target = train_target1[int(0.75 * train_target1.shape[0]) - seq_len:]

        data = pd.DataFrame(scale(data.values), columns=data.columns)
        data.index = target.index
        test_data = data[data.index >= '2016-04-21']
        test_target = target[target.index >= '2016-04-21']

        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target), valid_data,
                                   valid_target]

    return data_warehouse


def cnn_data_sequence_separately(tottal_data, tottal_target, data, target, seque_len):
    for index in range(data.shape[0] - seque_len + 1):
        tottal_data.append(data[index: index + seque_len])
        tottal_target.append(target[index + seque_len - 1])

    return tottal_data, tottal_target


def cnn_data_sequence(data_warehouse, seq_len):
    tottal_train_data = []
    tottal_train_target = []
    tottal_valid_data = []
    tottal_valid_target = []
    tottal_test_data = []
    tottal_test_target = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = cnn_data_sequence_separately(tottal_train_data, tottal_train_target,
                                                                              value[0], value[1], seq_len)
        tottal_test_data, tottal_test_target = cnn_data_sequence_separately(tottal_test_data, tottal_test_target,
                                                                            value[2], value[3], seq_len)
        tottal_valid_data, tottal_valid_target = cnn_data_sequence_separately(tottal_valid_data, tottal_valid_target,
                                                                              value[4], value[5], seq_len)

    tottal_train_data = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data = np.array(tottal_test_data)
    tottal_test_target = np.array(tottal_test_target)
    tottal_valid_data = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    tottal_train_data = tottal_train_data.reshape(tottal_train_data.shape[0], tottal_train_data.shape[1],
                                                  tottal_train_data.shape[2], 1)
    tottal_test_data = tottal_test_data.reshape(tottal_test_data.shape[0], tottal_test_data.shape[1],
                                                tottal_test_data.shape[2], 1)
    tottal_valid_data = tottal_valid_data.reshape(tottal_valid_data.shape[0], tottal_valid_data.shape[1],
                                                  tottal_valid_data.shape[2], 1)

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target


def sklearn_acc(model, test_data, test_target):
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro')]

    return acc_results


def train(data_warehouse, i):
    seq_len = 60
    epochs = 200
    drop = 0.1

    global cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target

    if i == 1:
        print('sequencing ...')
        cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
            data_warehouse, seq_len)

    my_file = Path(join(Base_dir,
        '2D-models/best-{}-{}-{}-{}-{}.h5'.format(epochs, seq_len, number_filter, drop, i)))
    filepath = join(Base_dir, '2D-models/best-{}-{}-{}-{}-{}.h5'.format(epochs, seq_len, number_filter, drop, i))
    if my_file.is_file():
        print('loading model')

    else:

        print(' fitting model to target')
        model = Sequential()
        #
        # layer 1
        model.add(
            Conv2D(number_filter[0], (1, number_feature), activation='relu', input_shape=(seq_len, number_feature, 1)))
        # layer 2
        model.add(Conv2D(number_filter[1], (3, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 1)))

        # layer 3
        model.add(Conv2D(number_filter[2], (3, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 1)))

        model.add(Flatten())
        model.add(Dropout(drop))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='Adam', loss='mae', metrics=['acc', f1])

        best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='max', period=1)


        model.fit(cnn_train_data, cnn_train_target, epochs=epochs, batch_size=128, verbose=1,
                        validation_data=(cnn_valid_data, cnn_valid_target), callbacks=[best_model])
    model = load_model(filepath, custom_objects={'f1': f1})

    return model, seq_len


def cnn_data_sequence_pre_train(data, target, seque_len):
    new_data = []
    new_target = []
    for index in range(data.shape[0] - seque_len + 1):
        new_data.append(data[index: index + seque_len])
        new_target.append(target[index + seque_len - 1])

    new_data = np.array(new_data)
    new_target = np.array(new_target)

    new_data = new_data.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2], 1)

    return new_data, new_target


def prediction(data_warehouse, model, seque_len, order_stocks, cnn_results):
    for name in order_stocks:
        value = data_warehouse[name]
        # train_data, train_target = cnn_data_sequence_pre_train(value[0], value[1], seque_len)
        test_data, test_target = cnn_data_sequence_pre_train(value[2], value[3], seque_len)
        # valid_data, valid_target = cnn_data_sequence_pre_train(value[4], value[5], seque_len)

        cnn_results.append(sklearn_acc(model, test_data, test_target)[2])

    return cnn_results


def run_cnn_ann(data_warehouse, order_stocks):
    cnn_results = []
    # dnn_results = []
    iterate_no = 4
    for i in range(1, iterate_no):
        K.clear_session()
        print(i)
        model, seq_len = train(data_warehouse, i)
        # cnn_results, dnn_results = prediction(data_warehouse, model, seq_len, order_stocks, cnn_results)
        cnn_results = prediction(data_warehouse, model, seq_len, order_stocks, cnn_results)

    cnn_results = np.array(cnn_results)
    cnn_results = cnn_results.reshape(iterate_no - 1, len(order_stocks))
    cnn_results = pd.DataFrame(cnn_results, columns=order_stocks)
    cnn_results = cnn_results.append([cnn_results.mean(), cnn_results.max(), cnn_results.std()], ignore_index=True)
    cnn_results.to_csv(join(Base_dir, '2D-models/new results.csv'), index=False)


Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = os.listdir(join(Base_dir, 'Dataset'))

# if moving average = 0 then we have no moving average
seq_len = 60
moving_average_day = 0
number_of_stocks = 0
number_feature = 0
samples_in_each_stock = 0
number_filter = [8, 8, 8]
predict_day = 1

cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = ([] for i in
                                                                                                      range(6))

print('Loading train data ...')
order_stocks = []
data_warehouse = costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)
# order_stocks = data_warehouse.keys()

print('number of stocks = '), number_of_stocks

run_cnn_ann(data_warehouse, order_stocks)









