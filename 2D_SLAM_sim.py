from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.random import randn, randint, uniform
import numpy.random
import numpy as np
import math
from filterpy.stats import plot_covariance_ellipse

seed = 0
numpy.random.seed(seed)

nT = 50 # Ideal number of targets

# Initialise camera position
x_pos = 0
y_pos = 0
y_vel = 2
bearing = 0.4

# Initialise target positions
targets = []
for i in range(nT):
    point = [uniform(-400,400), uniform(0, 800)]
    #if ((point[0] < -75) or (point[0] > 75)): # Remove targets that are too close to camera path
    targets.append(point)

#targets.insert(0,[50,50])
#targets.insert(0,[-50,50])
#targets.insert(0,[25,75])
nT = len(targets) # Update number of targets with removed targets

def XJacobian(x):
    # At state x return Jacobian matrix
    xp = float(x[0])
    yp = float(x[1])
    vp   = float(x[2])
    bear = float(x[3])

    nS = len(x) # Number of states
    nT = (len(x)-4)/2 # Number of targets
    out = np.zeros((int(nT), nS)) # Jacobian output array shape
    for i in range(int(nT)):
        tIndx = i*2+4 # First target element
        element = np.zeros(nS)
        targets_partial_deriv = array([(-float(x[tIndx+1]) + yp)/
                                ((float(x[tIndx]) - xp)**2 + (float(x[tIndx+1])
                                - yp)**2), (float(x[tIndx]) - xp)/
                                ((float(x[tIndx]) - xp)**2 + (float(x[tIndx+1])
                                - yp)**2)])
        
        element[tIndx:tIndx+2] = targets_partial_deriv
        element[0:4] = array ([-(-float(x[tIndx+1]) + yp)/
                               ((float(x[tIndx])- xp)**2 + (float(x[tIndx+1])
                                - yp)**2), -(float(x[tIndx]) - xp)/
                               ((float(x[tIndx]) - xp)**2 + (float(x[tIndx+1])
                                - yp)**2), 0, 1])
        out[i] = element
    return out

def hx(x):
    # Measurement expected for state x
    xp = float(x[0])
    yp = float(x[1])
    vp   = float(x[2])
    bear = float(x[3])

    nT = (len(x)-4)/2 # Number of targets
    out = []
    
    for i in range(int(nT)):
        tIndx = i*2+4 # First target element
        element = [math.atan2((float(x[tIndx+1] - yp)), (float(x[tIndx]) - xp))+bear]
        out.append(element)
    out = asarray(out)
    return out

def residual(a, b):
    y = a - b # calculate subtraction residual
    y = y % (2 * np.pi)    # move to 0 -> 2*pi
    y[y > np.pi] -= 2 * np.pi # rescale to -pi -> +pi
    return y

def update_camera(dt):
    global x_pos, y_pos, y_vel, bearing, targets
    # Process noise
    x_pos = x_pos  + .1 * randn()
    y_vel = y_vel + .01 * randn()
    bearing = bearing + .01 * randn()
    y_pos = y_pos + y_vel * dt

    measurements = []
    for T in targets:
        err = .05 * randn()
        ang = math.atan2((T[1]- y_pos), (T[0] - x_pos)) + bearing
        measurements.append([ang + err])
    measurements = asarray(measurements)

    return measurements

dt = 0.05

ekf = ExtendedKalmanFilter(dim_x = 4 + 2 * nT, dim_z = nT)

np_targets = asarray(targets) + 75 * randn(nT, 2)
#np_targets[0:3] = array([[25,75], [-50,50], [50,50]])
ekf.x = array([x_pos + 5, y_pos + 7, y_vel + .5, bearing + 1.5]) # Make incorrect guess
ekf.x = np.append(ekf.x, np_targets)
ekf.x = np.reshape(ekf.x, (1, 4 + 2 * nT)).T

ekf.F = eye(4 + 2 * nT)
ekf.F[1,2] = dt

range_var = 0.05 # Uncertainty of angle measurements
ekf.R = eye(nT) * range_var
ekf.Q = np.eye(4 + 2 * nT) * 1.000e-7 # Target position process covariance
ekf.Q[0,0], ekf.Q[1,1], ekf.Q[2,2], ekf.Q[3,3] = 1.000e-03, 1.0000e-06, 1.0000e-04, 1.0000e-5
ekf.P *= 1500 # Estimated accuracy of state estimate
#ekf.Q[4,4], ekf.Q[5,5], ekf.Q[6,6] = 1.0000e-10, 1.0000e-10, 1.0000e-10
#ekf.P[4,4], ekf.P[5,5], ekf.P[6,6] = 0, 0, 0

cameraTrk, predictionTrk, updateCovarTrk = [], [], []
MAng, TargetX, TargetX_Error = [], [], []
for i in range(int(300/dt)):
    z = update_camera(dt)
    cameraTrk.append((x_pos, y_pos, y_vel, bearing))
    MAng.append(z[0][0])

    TargetX.append((ekf.x[4:]))
    TargetX_Error.append(ekf.x[4:].flatten()-asarray(targets).flatten())

    # Update EKF with measurement, Jacobian function and function to find expected measurement
    ekf.update(z, XJacobian, hx, residual = residual) # Update predictions with measurements

    predictionTrk.append((ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3]))
    updateCovarTrk.append(ekf.P)
    
    ekf.predict() # Predict next state


## PLOT IT
cameraTrk = asarray(cameraTrk)
predictionTrk = asarray(predictionTrk)
updateCovarTrk = asarray(updateCovarTrk)

MAng = asarray(MAng)
TargetX = asarray(TargetX)
TargetX_Error = asarray(TargetX_Error)

xaxis = np.arange(0, 300, 0.05)
titles = ['X Position', 'Y Position', 'Y Velocity']
ylabels = ['X-Position (m)', 'Y-Position (m/s)', 'Y-Velocity (m)', 'Bearing (radians)']
fig, axs = plt.subplots(2,4, figsize = (50,5))
for plot in range(3):
    axs[0, plot].set_title(titles[plot])
    axs[0, plot].set_xlabel('Time (s)')
    axs[0, plot].set_ylabel(ylabels[plot])
    axs[0, plot].plot(xaxis, cameraTrk[:,plot], 'k')
    axs[0, plot].plot(xaxis, predictionTrk[:,plot], 'r')

axs[0, 3].set_title('Bearing')
axs[0, 3].set_xlabel('Time (s)')
axs[0, 3].set_ylabel('Bearing (radians)')
axs[0, 3].plot(xaxis, cameraTrk[:,3], 'k')
axs[0, 3].plot(xaxis, predictionTrk[:,3], 'r')

axs[1,0].set_title('Camera Trajectory (X position by Y position)')
axs[1, 0].set_xlabel('X-Position (m)')
axs[1, 0].set_ylabel('Y-Position (m)')
axs[1,0].plot(cameraTrk[:,0], cameraTrk[:,1], 'ko', markersize=1, markevery=5, alpha = .9)
axs[1,0].plot(predictionTrk[:,0], predictionTrk[:,1], 'ro', markersize=1, markevery=5, alpha = .9)

for i in range(nT):
    axs[1,0].plot(targets[i][0], targets[i][1], 'bx', markersize = 4, alpha = .7)

axs[1, 1].set_title('X Target Error')
axs[1, 2].set_title('Y Target Error')
axs[1, 3].set_title('X Target Locations')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 2].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('X-Position Error (m)')
axs[1, 2].set_ylabel('Y-Position Error (m)')

xavg = 0
yavg = 0
print(TargetX_Error.shape)
for i in range(0, nT*2, 2):
    axs[1,1].plot(xaxis, TargetX_Error[:,i])
    xavg +=TargetX_Error[5999,i]
    axs[1,2].plot(xaxis, TargetX_Error[:,i+1])
    yavg +=TargetX_Error[5999,i+1]

print(xavg/nT, yavg/nT)
## COVARIANCE ELLIPSES

cov_fig = plt.figure()
plt.plot(cameraTrk[:, 0], cameraTrk[:, 1], 'ko', markersize=1, markevery= 10, alpha = 0.3)
plt.title('Camera covariance plot (calibration)')
plt.xlabel('X-Position (m)')
plt.ylabel('Y-Position (m)')
for i in range(nT):
    plt.plot(targets[i][0], targets[i][1], 'bx', markersize = 4)
for i in range(0, predictionTrk.shape[0], 1000):
    plot_covariance_ellipse(
                        (predictionTrk[i, 0], predictionTrk[i, 1]), updateCovarTrk[i, 0:2, 0:2],
                        std=6, facecolor='g', alpha=0.8)

### ANIMATE

TargetX = np.reshape(TargetX, (TargetX.shape[0], nT, 2))[0::10]
cameraTrk = cameraTrk[0::10]
predictionTrk = predictionTrk[0::10]

fig_anim = plt.figure()
fig_anim.set_size_inches(5, 5, True)
ax_anim = plt.axes(xlim=(-500, 500), ylim=(0, 800))
ax_anim.set_title('Camera Trajectory (X position by Y position)')
ax_anim.set_xlabel('X-Position (m)')
ax_anim.set_ylabel('Y-Position (m)')

ground_truth, = ax_anim.plot(cameraTrk[0][0], cameraTrk[0][1], 'ko', markersize=1)
kalman_track, = ax_anim.plot(predictionTrk[0][0], predictionTrk[0][1], 'ro', markersize=1)
target_pred, = ax_anim.plot([], [], 'go', markersize = 4, alpha = .7)
lines = [plt.plot([], [], 'yo-', animated=True, markersize = 0)[0] for _ in range(nT)]
for i in range(nT):
    ax_anim.plot(targets[i][0], targets[i][1], 'bx', markersize = 4, alpha = .7)

def animate(i):
    ground_truth.set_alpha(0.4)
    kalman_track.set_alpha(0.4)
    ground_truth.set_data(cameraTrk[:i:,0], cameraTrk[:i:,1])
    kalman_track.set_data(predictionTrk[:i:,0], predictionTrk[:i:,1])
    target_pred.set_data(TargetX[i,:,0], TargetX[i,:,1])
    for j in range(nT):
        lines[j].set_data([TargetX[i,j,0], targets[j][0]], [TargetX[i,j,1], targets[j][1]])

anim = FuncAnimation(
    fig_anim, animate, interval=0, frames=cameraTrk.shape[0]-1)

path = 'D:\\Documents\\University\\MSc Project\\main\\Kalman\\videos\\'
anim.save(path + 'Seed {} ({} targets).mp4'.format(seed, nT), fps=60, bitrate = 1000)
plt.close()

print('Video Outputted.')
#plt.draw()
plt.show()