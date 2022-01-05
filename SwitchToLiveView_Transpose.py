import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.signal as signal
from scipy.stats import linregress

# %%
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

    print(fullData[11, :, :, 0])

    return fullData

# data =processData("2021-10-15_18-54-01_data_baseline")[3, :, :, 0:2500]

well_select = 23
data = processData("Jesses_DMD_Plate_1_Tissue")[:, :, :, 0:1000] #- processData("2021-10-15_18-54-01_data_baseline")[3, :, :, 0:2500]
pickle_in = open("Jesses_DMD_Plate_1.pickle", "rb")
algo = pickle.load(pickle_in)[0, 0:1000, :]
high_cut = 30 # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)
data = signal.filtfilt(b, a, data[:, 1, 0, :], axis=1)
# algo = signal.filtfilt(b, a, algo, axis=1)
data = np.transpose(data - np.amin(data, axis=1)[:, np.newaxis])
algo = algo - np.amin(algo, axis=0)[np.newaxis, :]



# Generate Heat Map with Slope and R sq values for all wells with more than 25 microns deflection
f_params = np.zeros((24, 3))

for i in range(0, 24):
    LinearFit_Model = linregress(data[:, i], algo[:, i])
    RegSlope = LinearFit_Model.slope
    RegIntercept = LinearFit_Model.intercept
    Rsq = (LinearFit_Model.rvalue) ** 2

    # print(RegSlope)
    # print(Rsq)
    # #
    # plt.plot(data[:, i], algo[:, i])
    # plt.plot(data[:, i], RegSlope * data[:, i] + RegIntercept)
    # # plt.plot(data[:, i], algo[:, i] - (RegSlope * data[:, i] + RegIntercept))
    # plt.ylabel("Algorithm x displacement output (mm)")
    # plt.xlabel("Flux density from sensor 2, x coordinate (mT)")
    # plt.grid(True)
    # plt.show()

    f_params[i, :] = np.array([RegSlope, Rsq, np.amax(algo[:, i] - (RegSlope * data[:, i] + RegIntercept))])

print(f_params)
f_params = f_params.reshape(6, 4, 3)
print(f_params)
goodFits = np.array(f_params)
goodFits[np.where(f_params[:, :, 1] < .95)] = -0*np.array([1, 1, 1])

# Distribution of voltages
fig, ax = plt.subplots()
im = ax.imshow(goodFits[:, :, 0])
# We want to show all ticks...
ax.set_xticks(np.arange(np.shape(f_params)[1]))
ax.set_yticks(np.arange(np.shape(f_params)[0]))
# ... and label them with the respective list entries
ax.set_xticklabels(["A", "B", "C", "D"])
ax.set_yticklabels(np.arange(6) + 1)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(6):
    for j in range(4):
        if f_params[i, j, 1] > .95:
            text = ax.text(j, i, "{0} \n {1} \n {2}".format(np.around(f_params[i, j, 0], 3),
                                                   np.around(f_params[i, j, 1], 3),
                                                     np.around(f_params[i, j, 2], 3)),
                           ha="center", va="center", color="k")
        else:
            text = ax.text(j, i, "{0} \n {1} \n {2}".format(np.around(f_params[i, j, 0], 3),
                                                   np.around(f_params[i, j, 1], 3),
                                                     np.around(f_params[i, j, 2], 3)),
                           ha="center", va="center", color="w")

plt.show()
