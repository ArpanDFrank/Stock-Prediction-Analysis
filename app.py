from keras.models import load_model
import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
from getData import *
from plot import *
import pandas as pd
from ml import *
from nextDayOpeningPrice import *
from ml2 import *
from ClosingPrice import *

st.title("Stock Trend Prediction")
user_input=st.text_input("Enter Stock Ticker","TSLA")

st.subheader(f"Data from 2010-01-01 - {dt.datetime.today().strftime('%Y-%m-%d')}")
st.write(getData(user_input).describe())


st.subheader("Closing Price vs Time chart")
fig,l1,l2 =graphs(user_input)
st.pyplot(fig)

ML_Model(user_input)
ML_Model2(user_input)
n1=predict_next_day_opening(user_input)
n2=predict_Closing_price(user_input,predict_next_day_opening(user_input))

if(l1.iloc[-1]<=l2.iloc[-1] or n2>n1):
    st.write("Probability of Upward Trend")
else:
    st.write("Probability of Downward Trend")




next_day_opening = n1
st.write("Predicted opening value for the next day: ",n1)
st.write("Predicted closing value for the following day: ",n2)



#x_train,y_train=generate_xtrain_ytrain(user_input)
model1 = load_model("keras_model.h5")
fig1 = model1EvalGraph(user_input,model1)
st.title("Model Prediction Evaluation for Opening Data")
st.pyplot(fig1)

model2=load_model("keras_model2.h5")
st.title("Model Prediction Evaluation for Closing Data")
fig2 = model2EvalGraph(user_input,model2)
st.pyplot(fig2)





