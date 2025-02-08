from calc import*
from getData import*
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

def get_traintestData(user):
    df=getData(user)
    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
    return data_training,data_testing

def scaling(user):
        (data_training,data_testing)=get_traintestData(user)
        scaler=MinMaxScaler(feature_range=(0,1))
        data_training=scaler.fit_transform(data_training)
        data_testing=scaler.fit_transform(data_testing)
        return data_training,data_testing


from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_xtrain_ytrain(user):
    data_trainingArray, data_testingArray = scaling(user)

    x_train = []
    y_train = []

    max_sequence_length = 100

    for i in range(max_sequence_length, data_trainingArray.shape[0]):
        sequence = data_trainingArray[i - max_sequence_length:i, 0]
        x_train.append(sequence)
        y_train.append(data_trainingArray[i, 0])

    x_train = pad_sequences(x_train, dtype=np.float32, padding='post', truncating='post')
    y_train = np.array(y_train, dtype=np.float32)

    return x_train, y_train

        
        
def ML_Model(user):
    (x_train,y_train)=generate_xtrain_ytrain(user)
    model=Sequential()
    
    model.add(LSTM(units=50,activation='tanh',return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=60,activation='tanh',return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(units=80,activation='tanh',return_sequences=True))
    model.add(Dropout(0.4))
    
    model.add(LSTM(units=120,activation='tanh'))
    model.add(Dropout(0.5))
    
    model.add(Dense(units=1))
    
    model.summary()
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(x_train,y_train,epochs=15, batch_size=32)
    model.save('keras_model.h5')
    
def generate_testData(user):
    (data_training, data_testing) = get_traintestData(user)
    #result = pd.concat([df1, df2], axis=0)
    final_df = pd.concat([data_training.tail(100),data_testing], axis=0,ignore_index=True)
    
    if final_df.empty:
        raise ValueError("DataFrame is empty")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(final_df)
    
    x_test = []
    y_test = []

    max_sequence_length = 100

    for i in range(max_sequence_length, input_data.shape[0]):
        sequence = input_data[i - max_sequence_length:i, 0]
        x_test.append(sequence)
        y_test.append(input_data[i, 0])

    x_test = pad_sequences(x_test, dtype=np.float32, padding='post', truncating='post')
    y_test = np.array(y_test, dtype=np.float32)
    
    scaler = scaler.scale_
    scale_factor = 1 / scaler
    
    return x_test, y_test, scale_factor