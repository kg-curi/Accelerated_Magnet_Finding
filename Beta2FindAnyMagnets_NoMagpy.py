# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pickle
import scipy.signal as signal
from numba import njit, jit, prange
import time


# def markPoint(event, x, y, flags, xyarray):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         xyarray.append([x, y])
#         print(xyarray)


# Simulate fields using a magnetic dipole model
# @njit(parallel=True) # Use numba to compile this method for large speed increase
# @jit(nopython=True)
# def meas_field(xpos, zpos, theta, ypos, phi, remn, arrays, manta):
#
#     fields = np.zeros((len(manta), 3))
#     theta = theta / 360 * 2 * np.pi
#     phi = phi / 360 * 2 * np.pi
#
#     for magnet in range(0, len(arrays)):
#         m = np.pi * (.75 / 2.0) ** 2 * remn[magnet] * np.asarray([np.sin(theta[magnet]) * np.cos(phi[magnet]),
#                                                                   np.sin(theta[magnet]) * np.sin(phi[magnet]),
#                                                                   np.cos(theta[magnet])])  # moment vectors
#         r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
#         rAbs = np.sqrt(np.sum(r ** 2, axis=1))
#
#         # simulate fields at sensors using dipole model for each magnet
#         for field in range(0, len(r)):
#             fields[field] += 3 * r[field] * np.dot(m, r[field]) / rAbs[field] ** 5 - m / rAbs[field] ** 3
#     print(fields)
#     return fields.reshape((1, 3*len(r)))[0] / 4 / np.pi

# Cost function
# @jit(nopython=True)
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
            fields[field] += 3 * r[field] * np.dot(m, r[field]) / rAbs[field] ** 5 - m / rAbs[field] ** 3

    # Bcalc = np.asarray(meas_field(x, z, theta, y, phi, remn, arrays, manta))


    return fields.reshape((1, 3*len(r)))[0] / 4 / np.pi - Bmeas


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

    # dummy = np.asarray([1])
    # meas_field(dummy, dummy, dummy, dummy, dummy, dummy, dummy)

    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)

    guess = [0, -5, 90, 1, 0, -575] #x, z, theta, y, phi remn
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

    triad = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

    manta = triad + arrays[0] // 6 * np.asarray([0, -19.5, 0]) + (arrays[0] % 6) * np.asarray([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + arrays[array] // 6 * np.asarray([0, -19.5, 0]) + (arrays[array] % 6) * np.asarray([19.5, 0, 0])), axis=0)

    Bmeas = np.zeros(len(arrays) * 9)

    print("start")

    for i in range(0, tp):  # 150

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes))) # Col 5

        if i == 1:
           x0 = np.asarray(res.x)

        res = least_squares(objective_function_ls, x0, args=(Bmeas, arrays, manta),
                            method='trf', ftol= 1e-4, verbose=2)


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
    fin = 200
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


# # Modify images to superimpose predicted coords for tracking visualization
#
# vidcap = cv2.VideoCapture('Dynamic Capture - Round 2 Video.avi')
# total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(total)
# for i in range(0, 23):
#     vidcap.read()
# success, input_image = vidcap.read()
# imsize = input_image.shape[0:2]
# Magnet_Position_Actual = [[554, 1108], [692, 1110], [703, 1139]]
#
# directory = r"C:\Users\NSB\PycharmProjects\pythonProject"
# os.chdir(directory)
#
# # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# # cv2.imshow("output", input_image)
# # cv2.setMouseCallback('output', markPoint, param=Magnet_Position_Actual)
# # cv2.waitKey(0)
#
# Scale = (Magnet_Position_Actual[1][0] - Magnet_Position_Actual[0][0]) / 3.2 # Pixels / mm
# XOrigin = float(Magnet_Position_Actual[2][0])
# ZOrigin = float(Magnet_Position_Actual[2][1])
#
# height = input_image.shape[0]
# x = posx[0] * Scale + XOrigin  # convert to pixels
# z = -posz[0] * Scale + ZOrigin
# cv2.rectangle(input_image, (int(x) - 10, int(z) - 10), (int(x) + 10, int(z) + 10), (255, 255, 255), -1)
# cv2.imwrite('frame_no0.jpg', input_image)
# print(x, z)
#
# for point in range(1, 375):
#     # NOTE First two points in Magnet_Position_Actual define the distance to be used as a scale, select left to right
#     # -- use the top vertex of the sensor (4 mm)
#     # Next point is reference point (x, y = 0)
#     # -- Somewhere towards the middle bottom of the sensor
#     vidcap.read()
#     success, input_image = vidcap.read()
#     if success:
#         print(point)
#         x = posx[point] * Scale + XOrigin # convert to pixels
#         z = -posz[point] * Scale + ZOrigin
#         cv2.rectangle(input_image, (int(x) - 10, int(z) - 10), (int(x) + 10, int(z) + 10), (255, 255, 255), -1)
#         cv2.namedWindow("output", cv2.WINDOW_NORMAL)
#         cv2.imshow("output", input_image)
#         cv2.waitKey(50)
#         framename = 'frame_no{0}.jpg'.format(point)
#         cv2.imwrite(framename, input_image)
#
# vidcap.release()
# cv2.destroyAllWindows()