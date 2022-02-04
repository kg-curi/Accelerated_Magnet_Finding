# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import _numdiff
import pickle
import scipy.signal as signal
from numba import njit, jit, prange
# from openpyxl import load_workbook
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
        fields += np.transpose((np.transpose(3 * r * np.dot(r, m)) / rAbs ** 5 - m / rAbs ** 3) / 4 / np.pi)


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

    guess = [0, -5, 90 / 360 * 2 * np.pi, 0, 0, -575] #x, z, theta, y, phi remn
    x0 = []
    for i in range(0, 6):
        for j in arrays:
            if i == 3:
                x0.append(guess[i] - 19.5 * (j %4))
            elif i == 0:
                x0.append(guess[i] + 19.5 * (j //4))
            else:
                x0.append(guess[i])
    print(x0)
    res = []
    tp = data.shape[3]
    start = time.time()

    triad = np.array([[-2.25, 2.25, 0], [2.25, 2.25, 0], [0, -2.25, 0]])  # sensor locations

    manta = triad + (arrays[0] %4) * np.array([0, -19.5, 0]) + (arrays[0] //4) * np.array([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + (arrays[array] %4) * np.array([0, -19.5, 0]) + (arrays[array] //4) * np.array([19.5, 0, 0])), axis=0)

    print(manta)

    Bmeas = np.zeros(len(arrays) * 9)

    print("start")

    for i in range(0, tp):  # 150

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes))) # Col 5

        if i >= 1:
           x0 = np.asarray(res.x)

        res = least_squares(objective_function_ls, x0, args=(Bmeas, arrays, manta),
                            method='lm', verbose=0, ftol=1e-1)
        jacobian = res.jac

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
    return np.array([xpos_est,
                     ypos_est,
                     zpos_est,
                     theta_est,
                     phi_est,
                     remn_est])

high_cut = 20  # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)

if  input("Regenerate Fields?"):

    pickle_in = open("Original.pickle", "rb")
    outputs1 = pickle.load(pickle_in)
    # pickle_in = open("Jesses_DMD_Plate_2.pickle", "rb")
    # outputs2 = pickle.load(pickle_in)

else:


    start = 0
    fin = 1000

    # Fields_baseline = signal.filtfilt(b, a, processData("Jesses_DMD_Plate_1_Baseline")[:, :, :, start:fin], axis=3)
    # Fields_tissue = signal.filtfilt(b, a, processData("Jesses_DMD_Plate_1_Tissue")[:, :, :, start:fin], axis=3)


    Fields_baseline = processData("B1_Variable_Stiffness_Plate_Baseline")[:, :, :, start:fin]
    Fields_tissue = processData("B1_Variable_Stiffness_Plate")[:, :, :, start:fin]

    Fields_baseline_avg = np.average(Fields_baseline, axis=3)

    Fields_s_BA = np.zeros(Fields_tissue.shape)
    for tp in range(0, len(Fields_s_BA[0, 0, 0, :])):
        Fields_s_BA[:, :, :, tp] = Fields_tissue[:, :, :, tp] - Fields_baseline_avg

    Fields_s_B = Fields_tissue - Fields_baseline

    outputs1 = getPositions(Fields_s_B)

    # pickle_out = open("Baseline_Avg_Sub_id.pickle", "wb")
    # pickle.dump(outputs1, pickle_out)
    # pickle_out.close()
    pickle_out = open("Original.pickle", "wb")
    pickle.dump(outputs1, pickle_out)
    pickle_out.close()

outputs1 = signal.filtfilt(b, a, outputs1, axis=1)
# outputs2 = signal.filtfilt(b, a, outputs2, axis=1)

#
#
# wb = load_workbook('Jesses_DMD_Plate_B5_1.xlsx')
# ws = wb.active
#
# # A1
# tcolumn = ws['A']
# ycolumn = ws['B']
#
# t = []
# x = []
# for timepoint in range(0, 700):
#     t.append(tcolumn[timepoint].value)
#     x.append(ycolumn[timepoint].value)
#
# x = np.array(x) / 1000
# x = x - np.amin(x)
# t = np.array(t)

# xalg = np.sqrt((outputs1[0, 0:1000, 17] - np.amin(outputs1[0, 0:1000, 17]))**2
#                + (-outputs1[1, 0:1000, 17] + outputs1[1, np.argmin(outputs1[0, 0:1000, 17]), 17])**2)
#
# xalg = signal.filtfilt(b, a, xalg)
#
# # xalg = outputs1[0, 0:2500, 17] - np.amin(outputs1[0, 0:2500, 17])
#
# peaks_alg, _ = signal.find_peaks(xalg, height=.15, distance=10)
# troughs_alg, _ = signal.find_peaks(-xalg, height= -.10, distance=50)
#
# print(np.amin(xalg))
#
# plt.plot(np.arange(0, outputs1.shape[1]/100, .01), xalg)
# plt.plot(np.arange(0, outputs1.shape[1]/100, .01,)[peaks_alg], xalg[peaks_alg], "x")
# plt.plot(np.arange(0, outputs1.shape[1]/100, .01,)[troughs_alg], xalg[troughs_alg], "x")
# plt.ylabel("Algorithm Magnet Position Change (mm)")
# plt.xlabel("Time Elapsed (s)")
# plt.show()

# twitch_amp = xalg[peaks_alg] - xalg[troughs_alg[1:len(troughs_alg)]]
# print(np.mean(twitch_amp))


#
# troughs_opt, _ = signal.find_peaks(-x, height= -.10, distance=10)
# peaks_opt, _ = signal.find_peaks(x, height= .10, distance=10)
#
# plt.plot(t, x)
# plt.plot(t[peaks_opt], x[peaks_opt], "x")
# plt.plot(t[troughs_opt], x[troughs_opt], "x")
# plt.ylabel("Optical Magnet Position Change (mm)")
# plt.xlabel("Time Elapsed (s)")
#
# twitch_amp = x[peaks_opt] - x[troughs_opt]
# print(np.mean(twitch_amp))
#
# plt.show()

# print(np.mean((outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]))[peaks]))
#
# print(np.std((outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]))[peaks]))


# print outputs
for i in range(0, len(outputs1)):
     print(outputs1[i, 0])

#Plot data
wells = ["A", "B", "C", "D"]
symbols = ["x", ".", "-", "v", ",", "s"]
nameMagnet = []
for i in range(0, outputs1.shape[2]):
    nameMagnet.append("{0}{1}".format(wells[i % 4], i // 4 + 1))
    plt.plot(np.arange(0, outputs1.shape[1] / 100, .01), outputs1[0, :, i]) #, symbols[i//4])

# x wrt t
# print(outputs.shape)
# plt.plot(np.arange(0, len(outputs[0, 0:1600, 22]) / 100, .01), outputs[0, 0:1600, 22] - np.amin(outputs[0, 0:1600, 22]))
# plt.plot(np.arange(0, len(outputs[0, 0:500, well_no]) / 100, .01), outputs[0, 0:500, well_no])
# plt.plot(np.arange(0, len(outputs[0, 0:500, well_no]) / 100, .01), outputs2[0, 0:500, well_no])

# plt.plot(np.arange(0, 5, .01,), outputs[0] [0:500, :])
# plt.plot(np.arange(0, outputs1.shape[1]/100, .01), outputs2[0, :, 2])
# plt.plot(np.arange(0, outputs2.shape[1]/100, .01,), outputs2[0, :, :])

plt.ylabel("predicted x position (mm)")
plt.xlabel("time elapsed (s)")
plt.grid(True)
plt.legend(nameMagnet)
plt.show()


print(outputs1)
# plt.plot(np.arange(0, 5, .01,), outputs[0] [0:500, :])
# plt.plot(np.arange(0, outputs.shape[1]/100, .01,), outputs[2, :, :])
plt.plot(outputs[0, :, 10], outputs[2, :, 10])

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

