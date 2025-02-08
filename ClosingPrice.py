from keras.models import load_model
import numpy as np
from ml2 import *

def predict_Closing_price(user,opening_price):
    model = load_model("keras_model2.h5")
    xtest,_,sf=generate_testData2(user)
    opening_price=opening_price/sf[0]
    xtest=xtest[-1]
    xtest = np.concatenate((xtest, [opening_price]))
    predicted_closing_price = model.predict(np.array([xtest]))
    return float(predicted_closing_price[0][0]*sf[0])

#print(predict_Closing_price("AAPL",120))