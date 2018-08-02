import pandas as pd
import sys
FILENAME = sys.argv[1]
data = pd.read_pickle(FILENAME+"_processed.pkl")
print(data)

print(len(data.loc[data['Label'] == 'normal']))
print(len(data.loc[data['Label'] == 'rtst']))

newx = 0
for i in data['Signal']:
	if len(i)!=1250:
		print(len(i))
		newx+=1
print("nx", newx)


print()
print("done")
print()