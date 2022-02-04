# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.signal as signal
from numba import njit
import time

MAGVOL = np.pi * (.75 / 2.0) ** 2 # for cylindrical magnet with diameter .75 mm and height 1 mm

# Compute moment vector of magnet
@njit(fastmath = True)
def compute_moment(thet,
                   ph,
                   rem):
    st = np.sin(thet)
    sph = np.sin(ph)
    ct = np.cos(thet)
    cph = np.cos(ph)
    return MAGVOL * rem * np.array([[st * cph], [st * sph], [ct]])  # moment vectors

# The jacobian is a matrix of partial derivatives of the cost function w.r.t. each parameter at each axis of each sensor
# used by a least squares algorithm to compute the derivative of the rms of the cost function for minimization.
# Built-in scipy least_squares computation of the jacobian is inefficient in the context of multi-magnet finding problems,
# scaling with M*(N)**2 where N is the number of magnets, for an M parameter fit.
# Computing it separately from least_squares with the following method lets it scale by M*(N)**1.
# This approach speeds up pure python alg ~30x, numba accelerated ~10x for 24 well plate with data from the beta 2.2
# The least_squares "method" should be set to "lm" and "ftol" to 1e-1 for specified performance
# The first 20 or so data points should be ignored, since there's a small transient
@njit(fastmath = True)
def compute_jacobian(pos,
                     Bmeas,
                     arrays,
                     manta,
                     eps):

    J = np.zeros((216, 144)) #Initialize jacobian matrix

    # x, z, theta y, phi, remn
    pos2 = pos.reshape(6, len(arrays))
    xpos = pos2[0]
    zpos = pos2[1]
    theta = pos2[2]
    ypos = pos2[3]
    phi = pos2[4]
    remn = pos2[5]

    # compute incremental changes in parameters
    rel_step = eps
    sign_x0 = np.zeros(len(pos))
    for x in range(0, len(pos)):
        if pos[x] >= 0:
            sign_x0[x] = 1
        else:
            sign_x0[x] = -1
    dx0 = rel_step * sign_x0 * np.maximum(1.0, np.abs(pos))

    # simulate fields at sensors using dipole model for each magnet
    # compute contributions from each magnet
    for magnet in range(0, len(arrays)):
        r = -np.array([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))
        m = compute_moment(theta[magnet], phi[magnet], remn[magnet])

        f0 = (np.transpose(3 * r * np.dot(r, m)) / rAbs ** 5 - m / rAbs ** 3) #dipole model

        # compute contributions from each parameter of each magnet
        for param in range(0, 6):
            rpert = r.copy()
            rAbspert = rAbs.copy()
            mpert = m.copy()
            pert = dx0[magnet + len(arrays) * param]
            perturbation_xyz = np.zeros((1, 3))
            perturbation_theta = theta[magnet]
            perturbation_phi = phi[magnet]
            perturbation_remn = remn[magnet]

            if param == 0 or param == 1 or param == 3:
                if param == 0:
                    perturbation_xyz[0, 0] = pert # x
                elif param == 1:
                    perturbation_xyz[0, 2] = pert # z
                elif param == 3:
                    perturbation_xyz[0, 1] = pert  # t
                rpert = r - perturbation_xyz #recompute r
                rAbspert = np.sqrt(np.sum(rpert ** 2, axis=1))
            else:
                if param == 2:
                    perturbation_theta += pert #phi
                elif param == 4:
                    perturbation_phi += pert #theta
                else:
                    perturbation_remn += pert #remn
                mpert = compute_moment(perturbation_theta, perturbation_phi, perturbation_remn)

            f1 = (np.transpose(3 * rpert * np.dot(rpert, mpert)) / rAbspert ** 5 - mpert / rAbspert ** 3)
            J[:, magnet + len(arrays) * param] = (np.transpose(f1 - f0) / 4 / np.pi /
                                                  dx0[magnet + param*len(arrays)]).copy().reshape((1, 3*len(r)))[0] # Assign output to column of jacobian
    return J




# Cost function
@njit(fastmath = True)
def objective_function_ls(pos,
                          Bmeas,
                          arrays,
                          manta,
                          eps):

    # x, z, theta y, phi, remn
    pos = pos.reshape(6, len(arrays))
    xpos = pos[0]
    zpos = pos[1]
    theta = pos[2]
    ypos = pos[3]
    phi = pos[4]
    remn = pos[5]

    fields = np.zeros((len(manta), 3))

    for magnet in range(0, len(arrays)):
        m = compute_moment(theta[magnet], phi[magnet], remn[magnet])

        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta  # radii to moment
        rAbs = np.sqrt(np.sum(r ** 2, axis=1))

        # simulate fields at sensors using dipole model for each magnet
        fields_from_magnet = (np.transpose(3 * r * np.dot(r, m)) / rAbs ** 5 - m / rAbs ** 3) / 4 / np.pi
        fields += np.transpose(fields_from_magnet)

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
    x0 = np.array(x0)
    res = []
    tp = data.shape[3]

    triad = np.array([[-2.25, 2.25, 0], [2.25, 2.25, 0], [0, -2.25, 0]])  # sensor locations

    manta = triad + (arrays[0] %4) * np.array([0, -19.5, 0]) + (arrays[0] //4) * np.array([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + (arrays[array] %4) * np.array([0, -19.5, 0]) + (arrays[array] //4) * np.array([19.5, 0, 0])), axis=0)

    Bmeas = np.zeros(len(arrays) * 9)

    eps = np.finfo(np.float64).eps**0.5 # machine epsilon for float64, calibrated for 2-point derivative calculation

    for j in range(0, len(arrays)):  # divide by how many columns active, should be divisible by 4
        Bmeas[j * 9:(j + 1) * 9] = np.asarray(data[arrays[j], :, :, 0].reshape((1, numSensors * numAxes)))  # Col 5

    dummy1 = objective_function_ls(x0, Bmeas, arrays, manta, eps) #compile accelerated methods
    dummy2 = compute_jacobian(x0, Bmeas, arrays, manta, eps)

    print("start")
    start = 0
    for i in range(0, tp):  # 150

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*9:(j+1) * 9] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes))) # Col 5

        if i >= 1:
            if i == 1:
                start = time.time() # time code after first datapoint
            x0 = np.array(res.x)

        res = least_squares(objective_function_ls,
                            x0,
                            args=(Bmeas, arrays, manta, eps),
                            method='lm',
                            verbose=0,
                            jac=compute_jacobian,
                            ftol=1e-1)

        outputs = np.asarray(res.x).reshape(6, len(arrays))
        xpos_est.append(outputs[0])
        ypos_est.append(outputs[3])
        zpos_est.append(outputs[1])
        theta_est.append(outputs[2])
        phi_est.append(outputs[4])
        remn_est.append(outputs[5])
        print(i)
    stop = time.time()
    print((start - stop)/100)
    return np.array([xpos_est,
                     ypos_est,
                     zpos_est,
                     theta_est,
                     phi_est,
                     remn_est])


### Feed Data To Algorithm and Handle Outputs ###

high_cut = 20  # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)

start = 0
fin = 1000

Fields_baseline = processData("B1_Variable_Stiffness_Plate_Baseline")[:, :, :, start:fin]
Fields_tissue = processData("B1_Variable_Stiffness_Plate")[:, :, :, start:fin]

Fields_baseline_avg = np.average(Fields_baseline, axis=3)

Fields_s_BA = np.zeros(Fields_tissue.shape)
for tp in range(0, len(Fields_s_BA[0, 0, 0, :])):
    Fields_s_BA[:, :, :, tp] = Fields_tissue[:, :, :, tp] - Fields_baseline_avg

outputs1 = getPositions(Fields_s_BA)

outputs1 = signal.filtfilt(b, a, outputs1, axis=1)

#Plot data
wells = ["A", "B", "C", "D"]
nameMagnet = []
for i in range(0, outputs1.shape[2]):
    nameMagnet.append("{0}{1}".format(wells[i % 4], i // 4 + 1))
    plt.plot(np.arange(0, outputs1.shape[1] / 100, .01), outputs1[0, :, i]) #, symbols[i//4])

plt.ylabel("predicted x position (mm)")
plt.xlabel("time elapsed (s)")
plt.grid(True)
plt.legend(nameMagnet)
plt.show()

