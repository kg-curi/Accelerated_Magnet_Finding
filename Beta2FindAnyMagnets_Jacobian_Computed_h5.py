# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.signal as signal
from numba import njit
import time
import h5py as h5

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
# The first 10 or so data points should be ignored, since there's a small transient
@njit(fastmath = True)
def compute_jacobian(pos,
                     Bmeas,
                     arrays,
                     manta,
                     eps,
                     num_sensors):

    J = np.zeros((3 * num_sensors * len(arrays), 6 * len(arrays))) #Initialize jacobian matrix

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
                          eps,
                          num_sensors):

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
    arrays = np.arange(24)
    wells = []
    memsicCenterOffset = 2 ** 15
    memsicMSB = 2 ** 16
    memsicFullScale = 16
    gauss2MilliTesla = .1

    rows = ["D", "C", "B", "A"]
    for array in arrays:
        wells.append("{0}{1}".format(rows[array % 4], len(arrays)//4 + 1 - (array // 4 + 1)))
    print(wells)
    datafile = h5.File("".join([dataName, wells[0], ".h5"]), 'r')
    tissue_data = (np.array(datafile['tissue_sensor_readings'], dtype=np.float64) - memsicCenterOffset) \
                  * memsicFullScale / memsicMSB * gauss2MilliTesla
    wells = wells[1:len(wells)]
    for well in wells:
        datafile = h5.File("".join([dataName, well, ".h5"]), 'r')
        tissue_data_well = np.array(datafile['tissue_sensor_readings'], dtype=np.float64)
        tissue_data = np.append(tissue_data, (tissue_data_well - memsicCenterOffset)
                                * memsicFullScale / memsicMSB * gauss2MilliTesla, axis=0)

    return tissue_data

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
    match = []

    arrays = []
    for array in range(0, 24):
        if ~np.isnan(data[0, 9 * array]):
            arrays.append(array)
    arrays = np.asarray(arrays)

    guess = [0, -5, np.radians(90), 2, 0, 1200] #x, z, theta, y, phi remn
    x0 = []
    for i in range(0, 6):
        for j in arrays:
            if i == 3:
                x0.append(guess[i] - 19.5 * (j % 4))
            elif i == 0:
                x0.append(guess[i] + 19.5 * (j //4))
            else:
                x0.append(guess[i])
    x0 = np.array(x0)
    print(x0)
    res = []
    tp = data.shape[1] - 1

    triad = np.array([[-2.25, 2.25, 0], [2.25, 2.25, 0], [0, -2.25, 0]])  # sensor locations

    manta = triad + (arrays[0] %4) * np.array([0, -19.5, 0]) + (arrays[0] //4) * np.array([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions
        manta = np.append(manta, (triad + (arrays[array] %4) * np.array([0, -19.5, 0]) + (arrays[array] //4) * np.array([19.5, 0, 0])), axis=0)

    eps = np.finfo(np.float64).eps**0.5 # machine epsilon for float64, calibrated for 2-point derivative calculation

    Bmeas = data[:, 0] # Col 5

    dummy1 = objective_function_ls(x0, Bmeas, arrays, manta, eps, numSensors) #compile accelerated methods
    dummy2 = compute_jacobian(x0, Bmeas, arrays, manta, eps, numSensors)

    print("start")
    start = 0
    for i in range(0, tp):  # 150

        Bmeas = data[:, i] # Col 5

        if i >= 1:
            if i == 1:
                start = time.time() # time code after first datapoint
            x0 = np.array(res.x)

        res = least_squares(objective_function_ls,
                            x0,
                            args=(Bmeas, arrays, manta, eps, numSensors),
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
dur = 93700

if input("Compute Positions"):
    # Fields_baseline = processData("C:\\Users\\NSB\\PycharmProjects\\pythonProject\\ML2022060200__2022_06_02_163405_iPSC supporting cells_run 3_Day 16_twitch\\Calibration__2022_06_02_162959__")
    # print("Processed 1")
    # Fields_tissue = processData("C:\\Users\\NSB\\PycharmProjects\\pythonProject\\ML2022060200__2022_06_02_164720_iPSC supporting cells_run 3_Day 16_force frequency\\ML2022060200__2022_06_02_164720__")
    # print("Processed 2")

    Fields_baseline = processData("C:\\Users\\NSB\\PycharmProjects\\pythonProject\\Accelerated_Magnet_Finding\\ML2022126006_Position 1 Baseline_2022_06_15_004655\\Calibration__2022_06_15_004304__")
    print("Processed 1")
    Fields_tissue = processData("C:\\Users\\NSB\\PycharmProjects\\pythonProject\\Accelerated_Magnet_Finding\\ML2022126006_Position 1 Baseline_2022_06_15_004655\\ML2022126006_Position 1 Baseline_2022_06_15_004655__")
    print("Processed 2")


    Fields_baseline_avg = np.average(Fields_baseline, axis=1)
    Fields_s_BA = (Fields_tissue.T - Fields_baseline_avg).T #Fields_baseline_avg
    # print(Fields_baseline_avg)
    # for i in range(0, 24):
    #     for j in range(0, 9):
    #         plt.plot(Fields_tissue[j + 9*i, :])
    #     plt.show()
    print("processed Full")
    outputs1 = getPositions(Fields_s_BA)
    np.save("Data.npy", outputs1)

else:
    outputs1 = np.load("Data.npy")[:, 0:dur]

high_cut = 30 # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)
outputs1 = signal.filtfilt(b, a, outputs1, axis=1)
print(outputs1.shape)
# plt.plot(np.diff(outputs1[0, :, :], axis=0))
well = 3
plt.plot(outputs1[0])

# plt.plot(outputs1[0, :, well] - np.min(outputs1[0, :, well]))
# plt.plot(outputs1[0, :, well + 4] - np.min(outputs1[0, :, well + 4]))
plt.show()

