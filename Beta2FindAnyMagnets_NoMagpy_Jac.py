# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pickle
import scipy.signal as signal
from numba import njit, jit, prange
import time



# Cost function
@njit(fastmath = True)
def objective_function_ls(pos, Bmeas, arrays, manta):
    # x, z, theta y, phi, remn
    pos = pos.reshape(6, len(arrays))
    xpos = pos[0]
    zpos = pos[1]
    theta = pos[2]
    ypos = pos[3]
    phi = pos[4]
    remn = pos[5]

    fields = np.zeros((len(manta), 3))
    magvol =  np.pi * (.75 / 2.0) ** 2


    # rx = -xpos[:, np.newaxis] + manta[:, 0]  # radii to moment
    # ry = -ypos[:, np.newaxis] + manta[:, 1]  # radii to moment
    # rz = -zpos[:, np.newaxis] + manta[:, 2]
    # r = np.array(np.transpose([rx, ry, rz]))
    # rAbs = np.sqrt(np.sum(r ** 2, axis=2))
    #
    # st = np.sin(theta)
    # sph = np.sin(phi)
    # ct = np.cos(theta)
    # cph = np.cos(phi)
    # m = magvol * remn * np.array([st * cph, st * sph, ct])  # moment vectors
    # test = r * np.dot(r, m)[:, :, 0] / (rAbs ** 5)
    # test3 = np.swapaxes(m[:, np.newaxis, :] / rAbs[:, :] ** 3, 1, 2)
    #
    # fields_from_magnet = (np.transpose(r) * np.transpose(np.dot(r, m)[:, :, 0]) / np.transpose(rAbs ** 5)
    #                       - np.swapaxes(m[:, np.newaxis, :] / rAbs[:, :] ** 3, 1, 2)) / 4 / np.pi
    #
    # fields = np.transpose(np.sum(fields_from_magnet, axis=1))

    for magnet in range(0, len(arrays)):
        st = np.sin(theta[magnet])
        sph = np.sin(phi[magnet])
        ct = np.cos(theta[magnet])
        cph = np.cos(phi[magnet])
        m = magvol * remn[magnet] * np.array([[st * cph], [st * sph], [ct]])  # moment vectors

        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))

        # simulate fields at sensors using dipole model for each magnet
        fields_from_magnet = (np.transpose(3 * r * np.dot(r, m)) / rAbs ** 5 - m / rAbs ** 3) / 4 / np.pi

        fields += np.transpose(fields_from_magnet)

    # return fields.reshape((1, 3*r.shape[0]))[0] - Bmeas

    return fields.reshape((1, 3*len(r)))[0] - Bmeas


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
    numSensors = 3
    numAxes = 3

    xpos_est = []
    ypos_est = []
    zpos_est = []
    theta_est = []
    phi_est = []
    remn_est = []

    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)
    arrays = np.asarray(arrays)

    guess = [0, -5, 60 / 360 * 2 * np.pi, 1, 0, -575] #x, z, theta, y, phi remn
    x0 = []
    for i in range(0, 6):
        for j in arrays:
            if i == 3:
                x0.append(guess[i] - 19.5 * (j // 6))
            elif i == 0:
                x0.append(guess[i] + 19.5 * (j % 6))
            else:
                x0.append(guess[i])

    res = []
    tp = data.shape[3]
    start = time.time()

    triad = np.array([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

    manta = triad + arrays[0] // 6 * np.array([0, -19.5, 0]) + (arrays[0] % 6) * np.array([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + arrays[array] // 6 * np.array([0, -19.5, 0]) + (arrays[array] % 6) * np.array([19.5, 0, 0])), axis=0)

    Bmeas = np.zeros(len(arrays) * 9)

    print("start")

    for i in range(0, tp):  # 150

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes))) # Col 5

        if i >= 1:
           x0 = np.asarray(res.x)

        res = least_squares(objective_function_ls, x0, args=(Bmeas, arrays, manta),
                            method='trf', ftol= 1e-2, verbose=0)
        # jacobian = res.jac

        outputs = np.asarray(res.x).reshape(6, len(arrays))
        xpos_est.append(outputs[0])
        ypos_est.append(outputs[3])
        zpos_est.append(outputs[1])
        theta_est.append(outputs[2])
        phi_est.append(outputs[4])
        remn_est.append(outputs[5])

        print(i)

    end = time.time()
    print("total processing time for {1} s of 24 well data= {0} s".format(end - start, tp / 100))
    return [np.asarray(xpos_est),
           np.asarray(ypos_est),
           np.asarray(zpos_est),
           np.asarray(theta_est),
           np.asarray(phi_est),
           np.asarray(remn_est)]


if  input("Regenerate Fields?"):

    pickle_in = open("Baseline_Avg_Sub.pickle", "rb")
    outputs1 = pickle.load(pickle_in)
    pickle_in = open("Baseline_Sub.pickle", "rb")
    outputs2 = pickle.load(pickle_in)




else:
    start = 0
    fin = 1000
    Fields_baseline = processData("2021-10-28_16-44-21_data_3rd round second run baseline")[:, :, :, start:fin]
    Fields_tissue = processData("2021-10-28_16-45-57_data_3rd round second with plate")[:, :, :, start:fin]

    Fields_baseline_avg = np.average(Fields_baseline, axis=3)

    Fields_s_BA = np.zeros(Fields_tissue.shape)
    for tp in range(0, len(Fields_s_BA[0, 0, 0, :])):
        Fields_s_BA[:, :, :, tp] = Fields_tissue[:, :, :, tp] - Fields_baseline_avg

    Fields_s_B = Fields_tissue - Fields_baseline

    outputs1 = getPositions(Fields_s_BA)
    outputs2 = getPositions(Fields_s_B)

    pickle_out = open("Baseline_Avg_Sub_id.pickle", "wb")
    pickle.dump(outputs1, pickle_out)
    pickle_out.close()
    pickle_out = open("Baseline_Sub.pickle", "wb")
    pickle.dump(outputs2, pickle_out)
    pickle_out.close()


high_cut = 30 # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)
outputs1 = signal.filtfilt(b, a, outputs1, axis=1)
outputs2 = signal.filtfilt(b, a, outputs2, axis=1)


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

# x wrt t
# print(outputs.shape)
# plt.plot(np.arange(0, len(outputs[0, 0:1600, 22]) / 100, .01), outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]))
# plt.plot(np.arange(0, len(outputs[0, 0:500, well_no]) / 100, .01), outputs[0, 0:500, well_no])
# plt.plot(np.arange(0, len(outputs[0, 0:500, well_no]) / 100, .01), outputs2[0, 0:500, well_no])

print(outputs1)
# plt.plot(np.arange(0, 5, .01,), outputs[0] [0:500, :])
plt.plot(np.arange(0, outputs1.shape[1]/100, .01,), outputs1[0, :, :])
# plt.plot(np.arange(0, outputs2.shape[1]/100, .01,), outputs2[0, :, :])

plt.ylabel("predicted x displacement (mm)")
plt.xlabel("time elapsed (s)")
plt.grid(True)
# plt.legend(nameMagnet)
plt.show()


print(outputs1)
# plt.plot(np.arange(0, 5, .01,), outputs[0] [0:500, :])
# plt.plot(np.arange(0, outputs.shape[1]/100, .01,), outputs[2, :, :])
plt.plot(outputs2[0, :, 10], outputs2[2, :, 10])

plt.ylabel("predicted x displacement (mm)")
plt.xlabel("predicted z displacement (mm)")
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
