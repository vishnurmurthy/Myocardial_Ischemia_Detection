import requests
import time

print('Starting Experiment')
# print(requests.get('http://localhost:22002/NeuLogAPI?GetSensorValue:[Pulse],[1]').text)
#requests.get('http://localhost:22002/NeuLogAPI?GetServerStatus')
start = time.time()
requests.get('http://localhost:22002/NeuLogAPI?StartExperiment:[Pulse],[1],[4],[5000]')
# print(requests.get('http://localhost:22002/NeuLogAPI?GetServerStatus').text)
print('Finished Experiment in ' + str(time.time() - start) + " seconds")
# print('Getting Data')
r =  requests.get('http://localhost:22002/NeuLogAPI?GetExperimentSamples')
requests.get('http://localhost:22002/NeuLogAPI?StartExperiment:[Pulse],[1],[4],[5000]')
print(r.text)
requests.get('http://localhost:22002/NeuLogAPI?StopExperiment')
 
