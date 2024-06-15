import numpy as np
import pickle

resolution = 0.1

x = pickle.load(open(str('area_points=' + str(3470000) + 'resolution=' + str(resolution) + '.p'), 'rb'))

for i in range(0, 38):
    print(i)
    area = pickle.load(open(str('Simulations/area_points_till_far=' + str(100000) + 'resolution=' + str(resolution) + 'try' + str(i) + '.p'), 'rb'))
    x = np.concatenate((x, area), axis = 1)

with open(str('area_points=' + str(len(x[1,:])) + 'resolution=' + str(resolution) + '.p'),  'wb') as f:
    pickle.dump(x, f, protocol=4)