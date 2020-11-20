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
    precision_neg = precision((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    f_posit = 2*((precision_pos*recall_pos)/(precision_pos+recall_pos+K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2

def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, parse_dates=['Date'])
        df_raw.index = df_raw['Date']
    except IOError:
        print("IO ERROR")
    return df_raw

def construct_data_warehouse(ROOT_PATH, file_names):
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    global predict_index
    global order_stocks
    tottal_train_data = np.empty((0,82))
    tottal_train_target = np.empty((0))
    tottal_test_data = np.empty((0,82))
    tottal_test_target = np.empty((0))

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
        del data['target']
        del data['Date']
        # data['Date'] = data['Date'].apply(lambda x: x.weekday())

        number_feature = data.shape[1]
        samples_in_each_stock = data.shape[0]

        train_data = data[data.index < '2016-04-21']
        train_data = scale(train_data)

        if df_name == predict_index:
            tottal_train_target = target[target.index < '2016-04-21']
            tottal_test_target = target[target.index >= '2016-04-21']

        data = pd.DataFrame(scale(data.values), columns=data.columns)
        data.index = target.index
        test_data = data[data.index >= '2016-04-21']

        tottal_train_data = np.concatenate((tottal_train_data, train_data))
        tottal_test_data = np.concatenate((tottal_test_data, test_data))

    train_size = int(tottal_train_data.shape[0]/number_of_stocks)
    test_size = int(tottal_test_data.shape[0] / number_of_stocks)
    tottal_train_data = tottal_train_data.reshape(number_of_stocks, train_size, number_feature)
    tottal_test_data = tottal_test_data.reshape(number_of_stocks, test_size, number_feature)


    return tottal_train_data, tottal_test_data, tottal_train_target, tottal_test_target

def cnn_data_sequence(data, target, seque_len):
    print ('sequencing data ...')
    new_train = []
    new_target = []

    for index in range(data.shape[1] - seque_len + 1):
        new_train.append(data[:, index: index + seque_len])
        new_target.append(target[index + seque_len - 1])

    new_train = np.array(new_train)
    new_target = np.array(new_target)

    return new_train, new_target

def sklearn_acc(model, test_data, test_target):
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro')]

    return acc_results

def CNN(train_data, test_data, train_target, test_target):
    # hisory of data in each sample
    seq_len = 60
    epoc = 100
    drop = 0.1

    # creating sample each containing #seq_len history
    cnn_train_data, cnn_train_target = cnn_data_sequence(train_data, train_target, seq_len)
    cnn_test_data, cnn_test_target = cnn_data_sequence(test_data, test_target, seq_len)
    result = []

    # Running CNNpred several times
    for i in range(1,40):
        K.clear_session()
        print ('i: ', i)
        my_file = Path( join(Base_dir, '3D-models/{}/model/{}-{}-{}-{}-{}.h5'.format(predict_index, epoc, seq_len, number_filter, drop, i)))
        filepath = join(Base_dir, '3D-models/{}/model/{}-{}-{}-{}-{}.h5'.format(predict_index, epoc, seq_len, number_filter, drop, i))

        # If the trained model doesn't exit, it is trained
        if my_file.is_file():
            print('loading model')

        else:
            print('fitting model')
            model = Sequential()

            #layer 1
            model.add(Conv2D(number_filter[0], (1, 1), activation='relu', input_shape=(number_of_stocks,seq_len, number_feature), data_format='channels_last'))
            #layer 2
            model.add(Conv2D(number_filter[1], (number_of_stocks, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(1, 2)))

            #layer 3
            model.add(Conv2D(number_filter[2], (1, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(1, 2)))

            model.add(Flatten())
            model.add(Dropout(drop))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer='Adam', loss='mae', metrics=['acc',f1])

            best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                                   save_weights_only=False, mode='max', period=1)

            model.fit(cnn_train_data, cnn_train_target, epochs=epoc, batch_size=128, verbose=0,callbacks=[best_model], validation_split=0.25)

        model = load_model(filepath, custom_objects={'f1': f1})
        test_pred = sklearn_acc(model,cnn_test_data, cnn_test_target)
        print (test_pred)
        result.append(test_pred)

    print('saving results')
    results = pd.DataFrame(result , columns=['MAE', 'Accuracy', 'F-score'])
    results = results.append([results.mean(), results.max(), results.std()], ignore_index=True)
    results.to_csv(join(Base_dir, '3D-models/{}/new results.csv'.format(predict_index)), index=False)


Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = os.listdir(join(Base_dir, 'Dataset'))

# if moving average = 0 then we have no moving average
moving_average_day = 0
number_of_stocks = 0
number_feature = 0
samples_in_each_stock = 0
number_filter = [8,8,8]
predict_day = 1
order_stocks = []
# Name of the index that is going to be predicted
predict_index = 'DJI'   # RUT, S&P, NYA, NASDAQ, DJI


print ('Loading train data ...')
train_data, test_data, train_target, test_target = construct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)
print ('number of stocks = ', number_of_stocks)
print ('fitting model')

CNN(train_data, test_data, train_target, test_target)










