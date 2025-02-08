from keras.models import load_model
import numpy as np
from ml import *

def predict_next_day_opening(user_input):

    model = load_model("keras_model.h5")

    x_test, _, scale_factor = generate_testData(user_input)

    predicted_prices_scaled = model.predict(x_test)

    predicted_prices = predicted_prices_scaled * scale_factor

    next_day_opening = predicted_prices[-1]

    return float(next_day_opening)

