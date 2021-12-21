# %% Import Libraries
import numpy as np
from scipy.optimize import least_squares
from numba import njit

# Simulate fields using a magnetic dipole model
# Use numba to compile this method for large speed increase
# takes lists of magnet info
# takes list showing which sensor arrays are active
# gets
# Cost function
@njit(parallel = True)
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

    return fields.reshape((1, 3*r.shape[0]))[0] - Bmeas


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


# Takes an array indexed as [well, sensor, axis, timepoint]
# Data should be the difference of the data with plate on the instrument and empty plate calibration data
# Assumes 3 active sensors for each well, that all active wells have magnets, and that all magnets have the well beneath them active
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

    triad = np.array([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

    manta = triad + arrays[0] // 6 * np.array([0, -19.5, 0]) + (arrays[0] % 6) * np.array([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + arrays[array] // 6 * np.array([0, -19.5, 0]) + (arrays[array] % 6) * np.array([19.5, 0, 0])), axis=0)

    Bmeas = np.zeros(len(arrays) * 9)

    print("start")

    for i in range(0, tp):  # 150

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes))) # Col 5

        if i == 1:
           x0 = np.asarray(res.x)

        res = least_squares(objective_function_ls, x0, args=(Bmeas, arrays, manta),
                            method='trf', ftol= 1e-2, verbose=0)


        outputs = np.asarray(res.x).reshape(6, len(arrays))
        xpos_est.append(outputs[0])
        ypos_est.append(outputs[3])
        zpos_est.append(outputs[1])
        theta_est.append(outputs[2])
        phi_est.append(outputs[4])
        remn_est.append(outputs[5])

    return [np.asarray(xpos_est),
            np.asarray(ypos_est),
            np.asarray(zpos_est),
            np.asarray(theta_est),
            np.asarray(phi_est),
            np.asarray(remn_est)]


