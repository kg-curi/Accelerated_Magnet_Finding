# %% Import Libraries
import numpy as np
from scipy.optimize import least_squares
from numba import jit

# Simulate fields using a magnetic dipole model
# Use numba to compile this method for large speed increase
# takes lists of magnet info
# takes list showing which sensor arrays are active
# gets
@jit(nopython=True)
def meas_field(xpos, zpos, theta, ypos, phi, remn, arrays):

    triad = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations relative to origin

    manta = triad + arrays[0] // 6 * np.asarray([0, -19.5, 0]) + (arrays[0] % 6) * np.asarray([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions unde each well-- probably not necessary to do each call
        manta = np.append(manta, (triad + arrays[array] // 6 * np.asarray([0, -19.5, 0]) + (arrays[array] % 6) * np.asarray([19.5, 0, 0])), axis=0)

    fields = np.zeros((len(manta), 3))
    theta = theta / 360 * 2 * np.pi # magnet pitch
    phi = phi / 360 * 2 * np.pi # magnet yaw

    for magnet in range(0, len(arrays)):
        m = np.pi * (.75 / 2.0) ** 2 * remn[magnet] * np.asarray([np.sin(theta[magnet]) * np.cos(phi[magnet]),
                                                                  np.sin(theta[magnet]) * np.sin(phi[magnet]),
                                                                  np.cos(theta[magnet])])  # compute moment vectors based on magnet strength and orientation
        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # compute distance vector from origin to moment
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))

        # simulate fields at sensors using dipole model for each magnet
        for field in range(0, len(r)):
            fields[field] += 3 * r[field] * np.dot(m, r[field]) / rAbs[field] ** 5 - m / rAbs[field] ** 3

    return fields.reshape((1, 3*len(r)))[0] / 4 / np.pi

# Cost function to be minimized by the least squares
def objective_function_ls(pos, Bmeas, arrays):
    # x, z, theta y, phi, remn
    pos = pos.reshape(6, len(arrays))
    x = pos[0]
    z = pos[1]
    theta = pos[2] # angle about y
    y = pos[3]
    phi = pos[4] # angle about z
    remn = pos[5] #remanence

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

    dummy = np.asarray([1])
    meas_field(dummy, dummy, dummy, dummy, dummy, dummy, dummy) #call meas_field once to compile it; there needs to be some delay before it's called again for it to compile

    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)
    print(arrays)

    guess = [0, -5, 95, 1, 0, -575] #guess for x, z, theta, y, phi, remanence
    x0 = []
    for i in range(0, 6): #make a guess for each active well
        for j in arrays:
            if i == 3:
                x0.append(guess[i] - 19.5 * (j // 6))
            elif i == 0:
                x0.append(guess[i] + 19.5 * (j % 6))
            else:
                x0.append(guess[i])
    print(x0)

    #run least squares on timepoint i
    res = []
    for i in range(0, 500):  # 150
        if len(res) > 0:
            x0 = np.asarray(res.x)

        increment = 1

        Bmeas = np.zeros(len(arrays) * 9)

        for j in range(0, len(arrays)):
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, increment * i].reshape((1, numSensors * numAxes))) #rearrange sensor readings as a 1d vector

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

    return [np.asarray(xpos_est),
           np.asarray(ypos_est),
           np.asarray(zpos_est),
           np.asarray(theta_est),
           np.asarray(phi_est),
           np.asarray(remn_est)]
