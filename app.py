import json
import numpy as np
import pickle
from flask import Flask, render_template, flash
import flask

app = Flask(__name__)

__locations = []
__data_columns = None
__model = None

app.config['SECRET_KEY'] = "my super secret"

@app.route('/', methods=['POST', 'GET'])
def predict_home_price():
  load_saved_artifacts()
  result = None
  if flask.request.method=='POST':
    try:
      location = flask.request.form['location']
      total_sqft = float(flask.request.form['total_sqft'])
      bath = int(flask.request.form['bath'])
      bhk = int(flask.request.form['bhk'])
      loc_index = __data_columns.index(location.lower())
    except:
      loc_index = -1
      flash("Please fill the form data correctly!")
      return render_template('home.html', result=result, __locations=__locations)
    
    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
      x[loc_index] = 1
    result = round(__model.predict([x])[0],2)
    
  return render_template('home.html', result=result, __locations=__locations)

def load_saved_artifacts():
  global __data_columns
  global __locations

  with open("./artifacts/columns.json", "r") as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]

  global __model
  if __model is None:
    with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
      __model = pickle.load(f)

  print("Loading saved artifacts")

if __name__ == "__main__":
  print("Server Running")
  app.run()