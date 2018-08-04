import requests
import time
import keras
import math
import numpy as np
import tensorflow as tf
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from selenium import webdriver

def collectData(seconds):
   # print('Starting Experiment')
   # start = time.time()
   # 50 samples per second * seconds
   samples = 50 * seconds
   requests.get('http://localhost:22002/NeuLogAPI?StartExperiment:[EKG],[1],[6],[' + str(samples) + ']')
   time.sleep(2 * seconds)
   # print('Finished Experiment in ' + str(time.time() - start) + " seconds")
   r =  requests.get('http://localhost:22002/NeuLogAPI?GetExperimentSamples')
   requests.get('http://localhost:22002/NeuLogAPI?StopExperiment')
   return r.text

if __name__ == "__main__":
   model = tf.keras.models.load_model("BWSI2018.h5")
   model.compile(loss='categorical_crossentropy', 
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                 metrics=['accuracy'])
   
   driver = webdriver.Chrome()
   while True:
      #seconds = input("Number of seconds to collect for: ")
      data = collectData(5)
      data = np.array(str(data).split('[')[2].split(']')[0].split(','))[2:]
      data = np.reshape(data, (1, len(data)))
      #if seconds=='0':
      #   data = np.random.randint(-10, 10, size = (1, 250))
      data = data.astype(np.float64)
      data[0] = data[0] - np.mean(data[0])
      data[0] = data[0] / np.std(data[0])
      pred = model.predict(data)
      dict = ['ST', 'RTST', 'NORMAL']
      trace = go.Scatter(x=np.arange(len(data[0])), y=data[0])
      plotly.offline.plot({"data": [trace]}, auto_open=False)
      driver.get("file:///C:/Users/Brian/Documents/MIT/Internships/MITLL/BWSI/Myocardial_Ischemia_Detection/scripts/temp-plot.html")
      # if math.isclose(0.4200246, list(pred[0])[0], rel_tol=0.001) and math.isclose(0.0834332, list(pred[0])[1], rel_tol=0.001):
      #   print('BAD MEASUREMENT')
      # else:
      print(pred)
      print("Predicted: " + dict[pred.argmax(axis=1)[0]])
