from ml import *
from sklearn.preprocessing import MinMaxScaler

def generate_testData():
    (data_training,data_testing)=get_traintestData()
    final_df=data_training.tail(100).append(data_testing,ignore_index=True)
    scaler=MinMaxScaler(feature_range=(0,1))
    input_data=scaler.fit_transform(final_df)
    x_test = []
    y_test = []

    max_sequence_length = 100

    for i in range(max_sequence_length, input_data.shape[0]):
        sequence = input_data[i - max_sequence_length:i, 0]
        x_test.append(sequence)
        y_test.append(input_data[i, 0])

    x_test = pad_sequences(x_test, dtype=np.float32, padding='post', truncating='post')
    y_test = np.array(y_test, dtype=np.float32)
    return x_test,y_test

