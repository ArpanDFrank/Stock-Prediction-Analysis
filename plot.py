from getData import *
from calc import *
import matplotlib.pyplot as plt
from ml import *

def graphs(user_input):
    df=getData(user_input)
    fig =plt.figure(figsize=(12,6))
    plt.plot(df.Close)  
    plt.plot(calc100_mean(user_input),'r')
    plt.plot(calc200_mean(user_input),'g')
    plt.xlabel("Time")
    plt.ylabel("Price")
    return fig,calc100_mean(user_input),calc200_mean(user_input)

def model1EvalGraph(user_input,model):
    df=getData(user_input)
    x_test,y_test,sf=generate_testData(user_input)
    ypredicted=model.predict(x_test)
    ypredicted=ypredicted*sf
    y_test=y_test*sf
    fig =plt.figure(figsize=(12,6))
    plt.plot(y_test,'r',label="Actual Values")
    plt.plot(ypredicted,'g',label="Predicted Values")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Model Prediction Evaluation for Opening Data")
    
    return fig

import matplotlib.pyplot as plt
from getData import getData
from ml2 import *
import numpy as np

def model2EvalGraph(user_input,model):
    #model = load_model("keras_model2.h5")
    x_test, y_test, scale_factor = generate_testData2(user_input)
    y_predicted = model.predict(x_test)
    print(scale_factor)
    y_predicted = y_predicted * scale_factor[0]
    y_test_scaled = y_test * scale_factor[0]
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test_scaled, 'r', label="Actual Values")
    plt.plot(y_predicted, 'g', label="Predicted Values")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Model Prediction Evaluation for Closing Data")
    return fig

