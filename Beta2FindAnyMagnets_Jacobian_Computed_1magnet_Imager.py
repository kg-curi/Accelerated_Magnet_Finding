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
def processData(dataName, Tissue):
    fileName = f'{dataName}.txt'
    numWells = 24
    numSensors = 3
    numAxes = 3
    memsicCenterOffset = 2 ** 15
    memsicMSB = 2 ** 16
    memsicFullScale = 16
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

    return fullData

# Generate initial guess data and run least squares optimizer on instrument data to get magnet positions
def getPositions(data):
    numSensors = data.shape[1]
    numAxes = data.shape[2]

    xpos_est = []
    ypos_est = []
    zpos_est = []
    theta_est = []
    phi_est = []
    remn_est = []
    match = []

    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)
    arrays = np.asarray(arrays)

    guess = [0, -5.5, np.radians(90), 2, 0, 1100] #x, z, theta, y, phi remn
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
    print(x0)
    res = []
    tp = data.shape[3]

    triad = np.array([[-2.25, 2.25, 0], [2.25, 2.25, 0], [0, -2.25, 0]])  # sensor locations

    manta = triad + (arrays[0] %4) * np.array([0, -19.5, 0]) + (arrays[0] //4) * np.array([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (triad + (arrays[array] %4) * np.array([0, -19.5, 0]) + (arrays[array] //4) * np.array([19.5, 0, 0])), axis=0)

    Bmeas = np.zeros(len(arrays) * numSensors * numAxes)

    eps = np.finfo(np.float64).eps**0.5 # machine epsilon for float64, calibrated for 2-point derivative calculation

    for j in range(0, len(arrays)):  # divide by how many columns active, should be divisible by 4
        Bmeas[j * numSensors * numAxes:(j + 1) * numSensors * numAxes] = np.asarray(data[arrays[j], :, :, 0].reshape((1, numSensors * numAxes)))  # Col 5

    dummy1 = objective_function_ls(x0, Bmeas, arrays, manta, eps, numSensors) #compile accelerated methods
    dummy2 = compute_jacobian(x0, Bmeas, arrays, manta, eps, numSensors)

    print("start")
    start = 0
    for i in range(0, tp):  # 150

        for j in range(0, len(arrays)): # divide by how many columns active, should be divisible by 4
            Bmeas[j*numSensors*numAxes:(j+1) * numSensors * numAxes] = np.asarray(data[arrays[j], :, :, i].reshape((1, numSensors * numAxes))) # Col 5

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
        # match.append((res.cost + Bmeas).reshape(24, 3, 3), i)
        # print(res.x)
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

high_cut = 30  # Hz
b, a = signal.butter(4, high_cut, 'low', fs=100)
mag_sel = 18
dur = 600000

fileid = "deflect2"

if input("Compute Positions"):
    Fields_baseline_full = processData("C:\\Users\\NSB\\PycharmProjects\\pythonProject\\mantarray-firmware-testcode\\Mantarray_Utility_Tool\\data\\deflect12_baseline", False)[:, :, :, :]
    print("Processed 1")
    Fields_tissue_full = processData("C:\\Users\\NSB\\PycharmProjects\\pythonProject\\mantarray-firmware-testcode\\Mantarray_Utility_Tool\\data\\{0}".format(fileid), True)[:, :, :, :]
    print("Processed 2")
    # Fields_baseline = np.zeros(Fields_baseline_full.shape)
    # Fields_tissue = np.zeros(Fields_tissue_full.shape)
    # Fields_baseline[mag_sel, :, :, :] = Fields_baseline_full[mag_sel, :, :, :]
    # Fields_tissue[mag_sel, :, :, :] = Fields_tissue_full[mag_sel, :, :, :]



    Fields_baseline_avg = np.average(Fields_baseline_full, axis=3)
    print(Fields_baseline_full.shape)
    Fields_s_BA = np.zeros(Fields_tissue_full.shape)
    for tp in range(0, len(Fields_s_BA[0, 0, 0, :])):
        Fields_s_BA[:, :, :, tp] = -(Fields_tissue_full[:, :, :, tp] - Fields_baseline_avg[:, :, :]) #Fields_baseline_avg

    for i in range(16, 17):
        for j in range (0, 3):
            plt.plot(Fields_baseline_full[i, 0, j, :])
            plt.plot(Fields_baseline_full[i, 1, j, :])
            plt.plot(Fields_baseline_full[i, 2, j, :])
    print(i)
    plt.show()

    print("processed Full")
    np.save("f{0}.npy".format(fileid), Fields_tissue_full)

    outputs1 = getPositions(Fields_s_BA[:, :, :, 0:dur])
    np.save("{0}.npy".format(fileid), outputs1)
    # #

else:
    outputs1 = np.load("outputs14.npy".format(fileid))[:, 0:dur]

    # outputs1 = np.load("{0}.npy".format(fileid))[:, 0:dur]
outputs1 = signal.filtfilt(b, a, outputs1, axis=1)
print(outputs1.shape)
# plt.plot(np.diff(outputs1[0, :, :], axis=0))
plt.plot(outputs1[0, :, :])
plt.show()


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(outputs1[0, :, :], outputs1[1, :, :], outputs1[2, :, :])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');


plt.show()

#Plot data
wells = ["A", "B", "C", "D"]
nameMagnet = []
t = np.arange(0, outputs1.shape[1] / 100, .01)

text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontfamily='monospace')
marker_style = dict(linestyle='none', color='0.8', markersize=10,
                    markerfacecolor="tab:blue", markeredgecolor="tab:blue")


for i in range(16, 17):
    nameMagnet.append("{0}{1}".format(wells[i % 4], i // 4 + 1))
    peaks, _ = signal.find_peaks(-np.diff(outputs1[0, :, i]), height=[.0045, .02], distance=100)
    # peaks = np.delete(peaks, [690, 721, 752])

    plt1 = plt.figure(1)

    plt.plot(t, outputs1[2, :, i])
    plt.plot(t[peaks + 100], outputs1[2, peaks + 100, i], "x")
    plt.xlabel("sample number")
    plt.ylabel("x pos (mm)")


    plt2 = plt.figure(2)

    plt.plot(t[0:len(t)-1], -np.diff(outputs1[0, :, i]))
    plt.plot(t[peaks], -np.diff(outputs1[0, :, i])[peaks], "x")
    plt.xlabel("sample number")
    plt.ylabel("x pos derivative (mm/sample)")

    # print(len(peaks))
    plt.show()


    marker = "${0}{1}$".format(wells[i % 4], i // 4 + 1)
    # peaks=peaks[0:len(peaks)-1]
    # ax.plot(np.diff(outputs1[0, peaks + 100, i]), marker=marker, **marker_style)
    print(len(peaks))

    # marker_style.update(markeredgecolor="none", markersize=10)
    # ax.plot(outputs1[0, peaks + 100, i], color="blue", label='X')
    # ax.plot(outputs1[1, peaks + 100, i], color="black", label='Y')
    # ax.plot(outputs1[2, peaks + 100, i], color="orange", label='Z')
    # ax.plot(np.diff(outputs1[0, peaks + 100, i]), color="blue", label='X')
    # ax.plot(np.diff(outputs1[1, peaks + 100, i]), color="black", label='Y')
    # ax.plot(np.diff(outputs1[2, peaks + 100, i]), color="orange", label='Z')
    # ax.plot(np.sqrt(np.sum(np.diff(outputs1[0:3, peaks + 100, i], axis=1)**2, axis=0)))


print(outputs1[5, peaks + 100, 16])
print(np.degrees(outputs1[4, peaks + 100, 16]))
print(np.degrees(outputs1[3, peaks + 100, 16]))

plt1 = plt.figure(1)
plt.plot(outputs1[0, peaks + 100, 16], color="blue", label='X')
plt.plot(outputs1[1, peaks + 100, 16], color="black", label='Y')
plt.plot(outputs1[2, peaks + 100, 16], color="orange", label='Z')
plt.legend()
plt.grid(True)

plt2 = plt.figure(2)
plt.plot(np.diff(outputs1[0, peaks + 100, 16]), color="blue", label='X')
plt.plot(np.diff(outputs1[1, peaks + 100, 16]), color="black", label='Y')
plt.plot(np.diff(outputs1[2, peaks + 100, 16]), color="orange", label='Z')
plt.ylabel("predicted position change (mm/movement)")
plt.xlabel("movement no.")
plt.legend()
plt.grid(True)
plt.show()




print(len(outputs1[0, peaks + 100, 0]))


# outputs_d = np.diff(outputs1[0, peaks + 100, mag_sel])
# peaks_d, _ = signal.find_peaks(outputs_d, height=3)
# outputs_clean = np.delete(np.diff(outputs1[0, peaks + 100, mag_sel]), peaks_d, None)
outputs_clean_resh = np.transpose(np.diff(outputs1[0, peaks + 100, mag_sel].reshape(19, 19)))
plt.show()
# plt.plot(np.transpose(outputs_clean_resh))
# plt.show()

plt.imshow(outputs_clean_resh + .5, aspect="auto")
plt.xlabel("y movement number")
plt.ylabel("x movement number")
plt.title("Algorithm x movement prediction error for .5 mm movement (percent of .5mm)")

for i in range(outputs_clean_resh.shape[0]):
    for j in range(outputs_clean_resh.shape[1]):
            text = plt.text(j, i, "{0}".format(np.around(100*(outputs_clean_resh[i, j] + .5) / .5, 0)),
                           ha="center", va="center", color="k", fontsize="8")
plt.show()

plt.plot(t, Fields_s_BA[0, 0, 0, 0:dur])
plt.plot(t[peaks + 100], Fields_s_BA[0, 0, 0, peaks+100], "x")
plt.show()

# field_map = np.transpose(np.sum(Fields_s_BA[mag_sel, 0:3, 0, peaks + 100], axis=1).reshape(38, 40))
# print(field_map)
# print(field_map.shape)
# plt.imshow(field_map, aspect="auto")
# plt.show()


fig, axs = plt.subplots(1, 3)
for i, ax in enumerate(axs):
    field_map = np.transpose(Fields_s_BA[mag_sel, i, 2, peaks + 100].reshape(38, 40))
    ax.imshow(field_map, aspect="auto")
plt.show()

x = np.arange(38)
y = np.arange(40)
X, Y = np.meshgrid(x, y)

field_map_x = np.transpose(Fields_s_BA[mag_sel, 0, 1, peaks + 100].reshape(38, 40))
field_map_y = np.transpose(Fields_s_BA[mag_sel, 0, 0, peaks + 100].reshape(38, 40))
plt.quiver(X, Y, field_map_x, field_map_y)
plt.show()

fig, axs = plt.subplots(1, 3)
for i, ax in enumerate(axs):
    field_map_x = np.transpose(Fields_s_BA[mag_sel, i, 1, peaks + 100].reshape(38, 40))
    field_map_y = np.transpose(Fields_s_BA[mag_sel, i, 0, peaks + 100].reshape(38, 40))
    ax.quiver(X, Y, field_map_x, field_map_y)
plt.show()

# plt.plot(np.ones(len(peaks)) * -.1)
# plt.plot(np.ones(len(peaks)) * -.09)
# plt.plot(np.ones(len(peaks)) * -.11)

# plt.plot(np.ones(len(peaks)) * 0)
#
#     # plt.plot(t[0:len(t) - 1], np.diff(outputs1[0, :, i])) #, symbols[i//4])
#     # plt.plot(peaks*.01, np.diff(outputs1[0, :, i])[peaks], 'x')
#
# plt.ylabel("predicted position change (mm/movement)")
# # plt.xlabel("time elapsed (s)")
# plt.xlabel("movement no.")
#
# plt.grid(True)
# # plt.legend(nameMagnet)
# plt.show()

