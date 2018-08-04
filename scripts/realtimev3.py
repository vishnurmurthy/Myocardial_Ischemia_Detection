import requests
import time
import keras
import tensorflow as tf

def collectData(seconds):
   # print('Starting Experiment')
   # start = time.time()
   # 50 samples per second * seconds
   samples = 50 * seconds
   requests.get('http://localhost:22002/NeuLogAPI?StartExperiment:[EKG],[1],[6],[' + str(samples) + ']')
   time.sleep(2 * seconds)
   # print('Finished Experiment in ' + str(time.time() - start) + " seconds")
   r =  requests.get('http://localhost:22002/NeuLogAPI?GetExperimentSamples')
   print(r.text)
   print(r.json)
   requests.get('http://localhost:22002/NeuLogAPI?StopExperiment')

if __name__ == "__main__":
   model = tf.keras.models.load_model("BWSI2018.h5")
   model.compile(loss='categorical_crossentropy', 
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                 metrics=['accuracy'])

   while True:
      seconds = input("Number of seconds to collect for: ")
      data = collectData(int(seconds))
      pred = model.predict(data)
      dict = ['ST', 'RTST', 'NORMAL']
      print("Predicted: " + dict[pred])
