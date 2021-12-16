import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from numba import jit

@jit(nopython=True)
def meas_field(xpos,
                  zpos,
                  theta,
                  ypos,
                  phi,
                  remn,
                  arrays,
                  sensors,
                  type):
    # neodymium magnet source (flexible post magnet)

    manta = sensors + arrays[0] // 6 * np.asarray([0, -19.5, 0]) + (arrays[0] % 6) * np.asarray([19.5, 0, 0])

    for array in range(1, len(arrays)): #Compute sensor positions -- probably not necessary to do each call
        manta = np.append(manta, (sensors + arrays[array] // 6 * np.asarray([0, -19.5, 0]) + (arrays[array] % 6) * np.asarray([19.5, 0, 0])), axis=0)

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

    if type == "diff":
        fields_diff = np.zeros((len(arrays), 2, 3))
        fields_resh = fields.reshape((len(arrays), 3, 3))  # axis, sensor, well)
        fields_diff[:, 0, :] = fields_resh[:, 1, :] - fields_resh[:, 0, :]
        fields_diff[:, 1, :] = fields_resh[:, 2, :] - fields_resh[:, 1, :]
        fields_resh = fields_diff.reshape((1, 2 * len(r)))[0]
    else:
        fields_resh = fields.reshape((1, 3*len(r)))[0]

    return fields_resh


def objective_function(pos,
                       Bmeas,
                       arrays,
                       sensors,
                       type):
    # x, z, theta y, phi, remn
    pos = pos.reshape(6, len(arrays))
    x = pos[0]
    z = pos[1]
    theta = pos[2]
    y = pos[3]
    phi = pos[4]
    remn = pos[5]

    Bcalc = np.asarray(meas_field(x, z, theta, y, phi, remn, arrays, sensors, type))

    return Bcalc - Bmeas


###### Run Least Squares Differential ######
def processData(data,
                sensors,
                type):

    numSensors = sensors.shape[0]
    numAxes = sensors.shape[1]

    xpos_est = np.zeros((MagNo, len(xmovement)))
    ypos_est = np.zeros((MagNo, len(xmovement)))
    zpos_est = np.zeros((MagNo, len(xmovement)))
    theta_est = np.zeros((MagNo, len(xmovement)))
    phi_est = np.zeros((MagNo, len(xmovement)))
    remn_est = np.zeros((MagNo, len(xmovement)))

    arrays = []
    for array in range(0, data.shape[0]):
        if data[array, 0, 0, 0]:
            arrays.append(array)

    guess = [0, -6, 95, 1, 0, 575]  # x, z, theta, y, phi remn
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

    for i in range(0, len(xmovement)):  # 150

        if type == "diff":

            data_diff = np.diff(data, axis=1)

            Bmeas = np.zeros(len(arrays) * numAxes * (numSensors - 1))

            for j in range(0, len(arrays)):  # divide by how many columns active, should be divisible by 4
                Bmeas[j * numAxes*(numSensors - 1):(j + 1) * numAxes*(numSensors - 1)] = np.asarray(
                    data_diff[arrays[j], :, :, i].reshape((1, (numSensors-1) * numAxes)))  # Col 5
        else:

            Bmeas = np.zeros(len(arrays) * numAxes * numSensors)

            for j in range(0, len(arrays)):  # divide by how many columns active, should be divisible by 4
                Bmeas[j * numAxes * numSensors:(j + 1) * numAxes * numSensors] = np.asarray(
                    data[arrays[j], :, :, i].reshape((1, numSensors * numAxes)))  # Col 5

        if len(res) > 0:
            x0 = np.asarray(res.x)

        res = least_squares(objective_function, x0, args=(Bmeas, arrays, sensors, type),
                            method='trf', ftol=1e-2)

        print(i)

        outputs = np.asarray(res.x).reshape(6, len(arrays))
        xpos_est[:, i] = outputs[0]
        ypos_est[:, i] = outputs[3]
        zpos_est[:, i] = outputs[1]
        theta_est[:, i] = outputs[2]
        phi_est[:, i] = outputs[4]
        remn_est[:, i] = outputs[5]

    return xpos_est


###### Generate Magnet Data ######
zh = 6
xsp = 19.5
ysp = 19.5


incr = .1
xmovement = []
for i in range(0, 20):
    for j in range(0, 4):
        xmovement.append(-incr * i)

MagNo = 24

magnets = []

for magnet in range(0, MagNo):
    M = magpy.source.magnet.Cylinder(mag=[0, 0, 600], dim=[.75, 1], pos=[19.5 * (magnet % 6), -19.5 * (magnet // 6), -zh], angle=90, axis=[0, 1, 0])
    magnets.append(M)

sensors = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])  # sensor locations

coll = magpy.Collection(magnets)
fields = np.zeros((len(magnets), 3, 3, len(xmovement)))

for pos in range(0,len(xmovement)):
    magnets[0].setPosition([xmovement[pos], 0, -zh])
    magnets[2].setPosition([xmovement[pos] + 19.5 * (2 % 6), 0, -zh])
    for mag in range(0, len(magnets)):
        noise = (np.random.rand(1, 3) - .5) * 5e-3
        for i in range(0, len(sensors)):
            noise2 = (np.random.rand(1, 3) - .5) * 2.5e-4
            fields[mag, i, :, pos] = coll.getB(sensors[i] + np.asarray([19.5 * (mag % 6), -19.5 * (mag // 6), 0])) + noise + noise2


xpos_est_diff = processData(fields, sensors, "diff")
xpos_est_se = processData(fields, sensors, "SE")

plt.plot(xpos_est_diff[1])
plt.plot(xpos_est_se[1])
# plt.plot(xmovement)
plt.xlabel("timepoint (arbitrary)")
plt.ylabel("magnet x position (mm)")
plt.legend(["predicted magnet position", "actual magnet position"])
plt.show()
