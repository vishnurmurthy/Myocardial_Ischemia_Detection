import requests
import time

def collectData(seconds):
   # print('Starting Experiment')
   # start = time.time()
   # 50 samples per second * seconds
   samples = 50 * seconds
   requests.get('http://localhost:22002/NeuLogAPI?StartExperiment:[Pulse],[1],[6],[' + str(samples) + ']')
   time.sleep(2 * seconds)
   # print('Finished Experiment in ' + str(time.time() - start) + " seconds")
   r =  requests.get('http://localhost:22002/NeuLogAPI?GetExperimentSamples')
   print(r.text)
   requests.get('http://localhost:22002/NeuLogAPI?StopExperiment')

if __name__ == "__main__":
   seconds = input("Number of seconds to collect for: ")
   collectData(int(seconds))
