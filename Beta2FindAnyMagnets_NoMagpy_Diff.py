# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pickle
import scipy.signal as signal
from numba import njit, jit, prange
import time


# Simulate fields using a magnetic dipole model
# @njit(parallel=True) # Use numba to compile this method for large speed increase
@jit(nopython=True)
def meas_field(xpos, zpos, theta, ypos, phi, remn, arrays):

    triad = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

    manta = triad + arrays[0] // 6 * np.asarray([0, -19.5, 0]) + (arrays[0] % 6) * np.asarray([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + arrays[array] // 6 * np.asarray([0, -19.5, 0]) + (arrays[array] % 6) * np.asarray([19.5, 0, 0])), axis=0)

    fields = np.zeros((len(manta), 3))
    theta = theta / 360 * 2 * np.pi
    phi = phi / 360 * 2 * np.pi

    for magnet in range(0, len(arrays)):
        m = np.pi * (.75 / 2.0) ** 2 * remn[magnet] * np.asarray([np.sin(theta[magnet]) * np.cos(phi[magnet]),
                                                                  np.sin(theta[magnet]) * np.sin(phi[magnet]),
                                                                  np.cos(theta[magnet])])  # moment vectors
        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))

        # simulate fields at sensors using dipole model for each magnet
        for field in range(0, len(r)):
            fields[field] += (3 * r[field] * np.dot(m, r[field]) / rAbs[field] ** 5 - m / rAbs[field] ** 3) / 4 / np.pi

    fields_diff = np.zeros((len(arrays), 2, 3))
    fields_resh = fields.reshape((len(arrays), 3, 3)) #axis, sensor, well)
    fields_diff[:, 0, :] = fields_resh[:, 1, :] - fields_resh[:, 0, :]
    fields_diff[:, 1, :] = fields_resh[:, 2, :] - fields_resh[:, 1, :]

    fields_diff_resh = fields_diff.reshape((1, 2*len(r)))[0]

    return fields_diff_resh

# Cost function
def objective_function_ls(pos, Bmeas, arrays):
    # x, z, theta y, phi, remn
    pos = pos.reshape(6, len(arrays))
    x = pos[0]
    z = pos[1]
    theta = pos[2]
    y = pos[3]
    phi = pos[4]
    remn = pos[5]

    Bcalc = np.asarray(meas_field(x, z, theta, y, phi, remn, arrays))

    return Bcalc - Bmeas


# Process data from instrument
def processData(dataName):
    fileName = f'{dataName}.txt'
    numWells = 24
    numSensors = 3
    numAxes = 3
    memsicCenterOffset = 2 ** 15
    memsicMSB = 2 ** 16
    memsicFullScale = 8
    gauss2MilliTesla = .1

    config = np.loadtxt(fileName, max_rows=1, delimiter=', ').reshape((numWells, numSensors, numAxes))

    activeSensors = np.any(config, axis=1)
    spacerCounter = 1
    timestampSpacer = [0]
    dataSpacer = []
    for (wellNum, sensorNum), status in np.ndenumerate(activeSensors):
        if status:
            numActiveAxes = np.count_nonzero(config[wellNum, sensorNum])
            for numAxis in range(1, numActiveAxes + 1):
                dataSpacer.append(timestampSpacer[spacerCounter - 1] + numAxis)
            timestampSpacer.append(timestampSpacer[spacerCounter - 1] + numActiveAxes + 1)
            spacerCounter += 1

    timestamps = np.loadtxt(fileName, skiprows=1, delimiter=', ', usecols=tuple(timestampSpacer[:-1])) / 1000000
    data = (np.loadtxt(fileName, skiprows=1, delimiter=', ',
                       usecols=tuple(dataSpacer)) - memsicCenterOffset) * memsicFullScale / memsicMSB * gauss2MilliTesla
    numSamples = timestamps.shape[0] - 2
    fullData = np.zeros((numWells, numSensors, numAxes, numSamples))
    fullTimestamps = np.zeros((numWells, numSensors, numSamples))

    dataCounter = 0
    for (wellNum, sensorNum, axisNum), status in np.ndenumerate(config):
        if status:
            fullData[wellNum, sensorNum, axisNum] = data[2:, dataCounter]
            dataCounter += 1

    timestampCounter = 0
    for (wellNum, sensorNum), status in np.ndenumerate(activeSensors):
        if status:
            fullTimestamps[wellNum, sensorNum] = timestamps[2:, timestampCounter]
            timestampCounter += 1

    outliers = np.argwhere(fullData < -.3)

    for outlier in outliers:
        fullData[outlier[0], outlier[1], outlier[2], outlier[3]] =\
            np.mean([fullData[outlier[0], outlier[1], outlier[2], outlier[3] - 1],
                     fullData[outlier[0], outlier[1], outlier[2], outlier[3] + 1]])

    return fullData

# Generate initial guess data and run least squares optimizer on instrument data to get magnet positions
def getPositions(data):
    numSensors = 2
    numAxes = 3

    xpos_est = []
    ypos_est = []
    zpos_est = []
    theta_est = []
    phi_est = []
    remn_est = []

    # dummy = np.asarray([1])
    # meas_field(dummy, dummy, dummy, dummy, dummy, dummy, dummy)

    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)
    print(arrays)

    guess = [0, -5, 95, 1, 0, -575] #x, z, theta, y, phi remn
    x0 = []
    for i in range(0, 6):
        for j in arrays:
            if i == 3:
                x0.append(guess[i] - 19.5 * (j // 6))
            elif i == 0:
                x0.append(guess[i] + 19.5 * (j % 6))
            else:
                x0.append(guess[i])
    print(x0)

    res = []
    start = time.time()
    for i in range(0, 2000):  # 150
        if len(res) > 0:
            x0 = np.asarray(res.x)

        increment = 1

        Bmeas = np.zeros(len(arrays) * 6)

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*6:(j+1) * 6] = np.asarray(data[arrays[j], :, :, increment * i].reshape((1, numSensors * numAxes))) # Col 5

        res = least_squares(objective_function_ls, x0, args=(Bmeas, arrays),
                            method='trf', ftol=1e-2)


        outputs = np.asarray(res.x).reshape(6, len(arrays))
        xpos_est.append(outputs[0])
        ypos_est.append(outputs[3])
        zpos_est.append(outputs[1])
        theta_est.append(outputs[2])
        phi_est.append(outputs[4])
        remn_est.append(outputs[5])

        print(i)

    end = time.time()
    print("total processing time for 20s of 24 well data= {0} s".format(end - start))
    return [np.asarray(xpos_est),
           np.asarray(ypos_est),
           np.asarray(zpos_est),
           np.asarray(theta_est),
           np.asarray(phi_est),
           np.asarray(remn_est)]


if  input("Regenerate Fields?"):

    pickle_in = open("Last_Data_Diff.pickle", "rb")
    outputs = pickle.load(pickle_in)
    # toplot = processData("2021-10-28_16-44-21_data_3rd round second run baseline")
    # for i in range(0, 9):
    #     plt.plot(toplot[2, i%3, i//3, :])
    # plt.show()




else:

    mtFields = np.diff(processData("2021-10-28_16-45-57_data_3rd round second with plate")[:, :, :, 0:2000], axis=1) - np.diff(processData("2021-10-28_16-44-21_data_3rd round second run baseline")[:, :, :, 0:2000], axis=1)

    print("Processed")
    outputs = getPositions(mtFields)

    pickle_out = open("Last_Data_Diff.pickle", "wb")
    pickle.dump(outputs, pickle_out)
    pickle_out.close()


high_cut = 30 # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)
# outputs = signal.filtfilt(b, a, outputs, axis=1)
# outputs2 = signal.filtfilt(b, a, outputs2, axis=1)

# peaks, _ = signal.find_peaks(outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]), height=.15)
# print((outputs[0, :, 22] - np.amin(outputs[0, :, 22]))[peaks])
#
# print(np.mean((outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]))[peaks]))
#
# print(np.std((outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]))[peaks]))

#Plot data
# nameMagnet = []
# for i in range(0, outputs.shape[2]):
#     nameMagnet.append("magnet {0}".format(i))

print(outputs)
plt.plot(np.arange(0, 20, .01,), outputs[0])
# plt.plot(outputs[0, 0:500, 0], outputs[0, 0:500, 23])
# plt.plot(outputs[0, 0:500, 0], outputs[0, 0:500, 1])

plt.ylabel("predicted x displacement (mm)")
plt.xlabel("time elapsed (s)")
plt.grid(True)
# plt.legend(nameMagnet)
plt.show()

# y wrt t
plt.plot(np.arange(0, len(outputs[0]) / 100, .01), outputs[1])
plt.ylabel("ypos")
plt.xlabel("datapoint number")
# plt.legend(nameMagnet)
plt.show()

#y wrt x
plt.plot(outputs[0], outputs[1], 'x')
plt.xlabel("predicted x position (mm)")
plt.ylabel("predicted y position (mm)")
# plt.legend(nameMagnet)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(150, -10)
plt.ylim(-70, 10)
plt.show()

# print outputs
for i in range(0, len(outputs)):
     print(outputs[i, 0])