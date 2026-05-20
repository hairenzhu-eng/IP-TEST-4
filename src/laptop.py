"""
Copyright (c) 2025 The uos_sess6072_build Authors.
Authors: Blair Thornton, Alec O'Loughlin, Miquel Massot
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np
import json
import os
from datetime import datetime
import argparse
import time
from pathlib import Path
import subprocess
import platform
import copy

from zeroros import Publisher, Subscriber
from zeroros.messages import String, Vector3, Vector3Stamped, Pose, PoseStamped, RBLaserScan
from zeroros.datalogger import DataLogger

from drivers.aruco import ArUcoUDPDriver
from drivers.rpi import Console, Rate
from drivers import __version__
from scipy.spatial.transform import Rotation as R

# ---------------- LiDAR + DBSCAN imports ----------------
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.cluster import DBSCAN

from model_sess6072 import TAM, Vehicle2D_e, dynamics_translation_e, dynamics_rotation_e, TrajectoryGenerate, RangeAngleKinematics
from math_sess6072 import l2m, HomogeneousTransformation, Vector, HomogeneousTransformation, Matrix, Identity
from model_sess6072 import rigid_body_kinematics # tried to remove <existing libraries>
from math_sess6072 import Inverse, Vector # tried to remove <existing libraries>

# enter additional library imports here

# define global variables 
N = 0
E = 1
G = 2
DOTN = 3
DOTE = 4
DOTG = 5

# define global functions
def get_wifi_name():
    result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "SSID" in line and "BSSID" not in line:
            return line.split(":")[1].strip()
    return 0

def Vector(dim): return np.zeros((dim, 1), dtype=float)

def rpm2N(x, fwd_lim = 2000, rev_lim = -2000): 
    tol = 10
    if x>fwd_lim: x=fwd_lim
    if x<rev_lim: x=rev_lim
    if abs(x) <= tol: return 0
    elif x>tol: return 1.541571428571430076E-7*x**2+3.293357142857142252E-4*x-1.401428571428424679E-3
    else: return -7.35749999999999954E-8*x**2+1.716749999999999581E-4*x-1.054478382732365536E-16

def N2rpm(x, fwd_lim = 1.2753, rev_lim = -0.63765): 
    tol = 10E-4
    if x>fwd_lim: x=fwd_lim
    if x<rev_lim: x=rev_lim    
    if abs(x) <= tol: return 0
    elif x>tol: return -5.727416623043567370E2*x**2+2.268233085499708977E3*x+2.958718669408357371E1
    else: return 2.397948698765131667E3*x**2+4.665568885752373717E3*x - -6.685183692128883879E-14

#global function for navigation
def extended_kalman_filter_predict(mu, Sigma, u, f, Q, dt):
    # (1) Project the state forward
    pred_mu, F = f(mu, u , dt)
      
    # (2) Project the error forward: 
    pred_Sigma = F@Sigma@F.T+Q
    
    # Return the predicted state and the covariance
    return pred_mu, pred_Sigma

def extended_kalman_filter_update(mu, Sigma, z, h, R, wrap_index = None):
    
    # Prepare the estimated measurement
    pred_z, H = h(mu)
 
    # (3) Compute the Kalman gain
    K = Sigma@ H.T@ Inverse(H@Sigma@H.T + R)
    
    # (4) Compute the updated state estimate
    delta_z = z- pred_z        
    if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
    cor_mu = mu + K@(delta_z)

    # (5) Compute the updated state covariance
    cor_Sigma = (Identity(mu.shape[0]) - K @ H) @ Sigma
    
    # Return the state and the covariance
    return cor_mu, cor_Sigma

def h_pose_update(x):
    est_measurement = Vector(6)
    est_measurement[N] = x[N]
    est_measurement[E] = x[E]
    est_measurement[G] = x[G]
    H = Matrix(6,6)
    H[N, N] = 1
    H[E, E] = 1
    H[G, G] = 1
    return est_measurement, H

def h_grate_update(x):
    est_measurement = Vector(6)
    est_measurement[DOTG] = x[DOTG]

    H=Matrix(6,6)
    H[DOTG,DOTG]=1
    return est_measurement, H 

# main class
class LaptopController:
    def __init__(self, OPERATING_MODE):
        
        ########### DEFINE ARUCO MARKER ID ###################                     
        MARKER_ID = 24 # <<< CHANGE TO YOUR ROBOT'S ARUCO ID

        ########### SET NETWORK CONDITIONS ###################             
        if OPERATING_MODE != 2: # robot
            self.robot_ip = "192.168.10.1"
            self.robot_available = False
            self.sim_init = False
            aruco_params = {
                "port": 50001,  # Port to listen to (DO NOT CHANGE)
                "marker_id": MARKER_ID,  # Marker ID to listen to
            }                     
            if platform.system() == "Windows": wifi_name = get_wifi_name()
            else: wifi_name = "SmartCatXX"

        elif OPERATING_MODE == 2: # webots
            self.robot_ip = "127.0.0.1"          
            aruco_params = {
                "port": 50000,  # Port to listen to (DO NOT CHANGE)
                "marker_id": 0,  # Overide for WEBOTS (DO NOT CHANGE)
            }
            wifi_name = "WEBOTS"
            self.sim_init = True # Deal with webots timestamps

        self.sim_time_offset = 0.0
                            
        Console.info("Connecting to:", self.robot_ip, "")
        if wifi_name:            
            Console.info(f"You are connected to {wifi_name}")
        else:
            Console.info("No WiFi connection detected")

        # store operating mode
        self.OPERATING_MODE = OPERATING_MODE

        ########### INITIALISE DATA LOGS ###################                     
        filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = Path("logs/log_" + filename_time + ".csv")
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        with self.filename.open('w') as f:
            f.write("EpochTime(s),TimeFromStart(s),right_prop_rate(rad/s),left_prop_rate(rad/s),LastDT(s),Yaw(rad),North(m),East(m),IMUSensedYawRate(rad/s),IMUIntegratedYaw(rad),IMUSensedTimeStamp(s),ARUCOSensedNorth(m),ARUCOSensedEast(m),ARUCOSensedYaw(rad),ArucoSensedTimeStamp(s),DepthTimeStamp(s),Depth(m)\n")      
        global file 
        file = self.filename

        ########### ENTER WAYPOINT VARIABLES ###############
        # Start waypoint: (North, East) in metres
        start_north, start_east = 0, 1

        # Goal waypoint: (North, East) in metres
        goal_north, goal_east = 10, 1

        north_path = [start_north, goal_north]
        east_path = [start_east, goal_east]
        
        self.waypoints = []
        
        for i in range(len(north_path)):
            waypoint = Vector3()
            
            waypoint.y = north_path[i]
            waypoint.x = east_path[i]
            
            self.waypoints.append(waypoint)
            print("WAYPOINTS: ", self.waypoints)
            print("WAYPOINTS TYPE: ", type(self.waypoints))
        
        ########### INITIALISE ROBOT VARIABLES #############        
        rate = 5.0  # Hz
        self.r = Rate(rate)
        self.lastdt = 1/rate        
        self.starttime = time.time()
        self.timefromstart = None
        self.prev_sensed = None # originally None
        
        self.sensed_imu_yaw_rate_rad_s = None
        self.sensed_imu_stamp_s = None
        self.sensed_imu_prev_stamp_s = None
        self.sensed_yaw_rate = None
        self.integrated_yaw = 0

        self.sensed_pos_northings_m = None
        self.sensed_pos_eastings_m = None
        self.sensed_pos_yaw_rad = None
        self.sensed_pos_stamp_s = None
        self.sensed_bottom_depth_m = None        
        self.sensed_bottom_depth_stamp_s = None        

        # ---------------- LiDAR definitions ----------------
        self.lidar_data = None
        self.lidar_data_rb = None
        self.lidar_timestamp_s = None
        self.latest_lidar_received_s = None
        self.lidar_new = False
        self.lidar_x_bl = 0.1
        self.lidar_y_bl = 0.0
        self.lidar_gamma_bl = 0.0
        self.lidar = RangeAngleKinematics(self.lidar_x_bl, self.lidar_y_bl, self.lidar_gamma_bl)

        # ---------------- LiDAR DBSCAN parameters and outputs ----------------
        self.lidar_dbscan_eps_m = 0.20
        self.lidar_dbscan_min_points = 3
        self.lidar_points_body = np.empty((0, 2))
        self.lidar_cluster_labels = np.array([], dtype=int)
        self.lidar_obstacles = []
        self.lidar_obstacle_centres_body = np.empty((0, 2))
        self.lidar_obstacle_centres_ne = np.empty((0, 2))
        self.lidar_obstacle_distances_m = np.array([])
        self.lidar_obstacle_angles_rad = np.array([])
        self.nearest_lidar_obstacle = None

        # ---------------- LiDAR sector parameters ----------------
        self.front_angle_limit = np.deg2rad(25)
        self.front_block_threshold = 0.5
        self.min_front_close_beams = 5
        self.side_angle_min = np.deg2rad(45)
        self.side_angle_max = np.deg2rad(120)

        self.map_observation_class = "unknown"
        self.front_clearance_m = np.inf
        self.left_clearance_m = np.inf
        self.right_clearance_m = np.inf
        
        self.initial_state = Vector(6)
        self.initial_state[N] = start_north
        self.initial_state[E] = start_east
        self.initial_state[G] = 0
        self.initial_state[DOTN] = 0
        self.initial_state[DOTE] = 0
        self.initial_state[DOTG] = 0
        
        self.North = self.initial_state[N][0]
        self.East = self.initial_state[E][0]
        self.Yaw = self.initial_state[G][0]         

        self.right_rate = 0
        self.left_rate = 0

        ############################# MOTION MODEL VARIABLES #######################
        # Body-force model: positive force from either thruster acts forward.
        # The right propeller command sign is handled at the RPM conversion.
        phi=l2m([0,0])        
        x=l2m([0,0])
        # Body-frame lateral offsets: right thruster is negative y, left is positive y.
        y=l2m([-0.09,0.09])
        
        self.G=TAM(phi,x,y)
        print('G = ',self.G)
        
        # hull, water properties
        rho = 1000 # density of water in kg/m3
        draft = 0.07 #m
        beam = 0.04 #m of the immersed hull section
        length = 0.5 #m
        width = 0.4 #m # of the whole hull

        # from ESDU 71016. Fluid forces, pressures and moments on rectangular blocks. ESDU 71016 ESDU International, London
        CD = 7#1.5 # approximation for block from Newman (0.9 to 2.75) 
        A = 2*beam*draft #catamaran cross section in surge
        k_drag = 0.5*rho*CD*A

        # Added mass from Imlay 1961, Technical Report DTMB - assuming a prolate spheroid
        mass = 3
        e = 1 - (beam/length)**2
        alpha = (2*(1-e**2)/e**3)*(0.5*np.log((1+e)/(1-e))-e) #note np.log() = ln(), np.log10()=log()

        mass_add = 2*alpha*mass/(2-alpha)  # kg of water pushed by hull with, note this is for an infinite

        m_tot = mass + mass_add

        # Added mass from Imlay 1961, Technical Report DTMB - assuming a prolate spheroid
        I_66 = mass*((length/2)**2+(width/2)**2)/4 # rough approximation as rectangle
        e = 1 - (beam/length)**2
        alpha = (2*(1-e**2)/e**3)*(0.5*np.log((1+e)/(1-e))-e) #note np.log() = ln(), np.log10()=log()
        beta = 1/e**2 - ((1-e**2)/(2*e**3)) * np.log((1+e)/(1-e))  # kg of water pushed by hull with, note this is for an infinite

        I66_add = 2*(1/5)*mass*((draft**2-length**2)**2*(alpha-beta)/(2*(draft**2-length**2)+(draft**2+length**2)/(beta-alpha)))

        I_tot = I_66+I66_add

        # drag B_66
        B_66 = 0.12#0.12
        
        self.initial_pose = True # Set false after pose is initialised
        

        # read these into our vehicle class
        self.robot = Vehicle2D_e(m_tot,I_tot,k_drag,B_66)
        self.robot.info()
        
        self.v_robot = Vector(3) # initially stationary velocity vector in e frame
        self.p_robot = Vector(3); self.p_robot[0] = start_north; self.p_robot[1] = start_east; self.p_robot[2] = np.deg2rad(0) # pose in the e frame
        
        ############################# CONTROL VARIABLES #######################
        # Setup control parameters
        #################################################################
        tau_s = 2 #s to remove along track error # 0.5
        self.L = 0.3#m distance to remove normal and angular error
        self.ks =  1/tau_s
        self.kn = None 
        self.kg = None
        
        self.v_max = 0.25 #fastest the robot can go # 0.2
        self.w_max = np.deg2rad(30) #fastest the robot can turn # 30
        # setup a contranor to store controls
        self.U = Vector(2).T
        ################################################################
        # Setup trajectory
        #################################################################
        v = 0.1 # 0.1 
        a = 0.4 # 0.1 
        self.s = TrajectoryGenerate(north_path,east_path)
        self.s.path_to_trajectory(v, a)

        # Generate turning arcs trajectory
        self.arc_radius = 0.02
        self.s.turning_arcs(self.arc_radius) 
        self.s.wp_id = len(self.s.P_arc) - 1
        self.trajectory_duration_s = float(self.s.Tp_arc[-1][0])

        ############################# EKF VARIABLES ####################
        # State x = [N, E, G, Ndot, Edot, Gdot]^T
        self.mu = Vector(6)
        self.mu[N]    = self.initial_state[N]
        self.mu[E]    = self.initial_state[E]
        self.mu[G]    = self.initial_state[G]
        self.mu[DOTN] = self.initial_state[DOTN]
        self.mu[DOTE] = self.initial_state[DOTE]
        self.mu[DOTG] = self.initial_state[DOTG]
        
        # Initial covariance
        self.Sigma = Identity(6)
        # Position uncertainty (m^2)
        self.Sigma[N, N]   = 0.01      # 0.1 m std
        self.Sigma[E, E]   = 0.01
        # Heading uncertainty (rad^2)
        self.Sigma[G, G]   = np.deg2rad(5.0)**2
        # Velocity uncertainty ((m/s)^2 and (rad/s)^2)
        self.Sigma[DOTN, DOTN] = 0.01
        self.Sigma[DOTE, DOTE] = 0.01
        self.Sigma[DOTG, DOTG] = np.deg2rad(10.0)**2
        
        # Process noise Q (very simple diagonal)
        self.Q = Identity(6)
        q_pos = 1e-4
        q_vel = 1e-3
        self.Q[N, N]   = q_pos
        self.Q[E, E]   = q_pos
        self.Q[G, G]   = 1e-5
        self.Q[DOTN, DOTN] = q_vel
        self.Q[DOTE, DOTE] = q_vel
        self.Q[DOTG, DOTG] = 1e-4
        
        # Measurement noise for ArUco pose (N, E, G)
        self.R_pose = Identity(6)
        self.R_pose[N, N] = 0.02**2                 # 2 cm std
        self.R_pose[E, E] = 0.02**2
        self.R_pose[G, G] = np.deg2rad(2.0)**2      # 2 deg std
        
        # Measurement noise for IMU yaw rate (Gdot)
        self.R_grate = Identity(6)
        self.R_grate[DOTG, DOTG] = np.deg2rad(1.0)**2
        
        # Time bookkeeping for EKF (not strictly needed, but handy)
        self.last_nav_t = self.starttime
                    
        ############################# DECLARE PUBLISHERS AND SUBSCRIBERS ######         
        self.control_pub = Publisher("/control", Vector3, ip=self.robot_ip)
        self.config_pub = Publisher("/config", String, ip=self.robot_ip)
        self.imu_sub = Subscriber("/imu", Vector3, self.imu_cb, ip=self.robot_ip)
        self.sonar_sub = Subscriber("/sonar", Vector3, self.sonar_cb, ip=self.robot_ip)
        self.lidar_sub = Subscriber("/lidar", RBLaserScan, self.lidar_callback, ip=self.robot_ip)
        self.console_sub = Subscriber("/command", String, self.command_cb, ip=self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)        
        # a callback only used by WEBOTS to fake Aruco readings 
        self.groundtruth_sub = Subscriber("/groundtruth", PoseStamped, self.groundtruth_callback, ip=self.robot_ip) 
        self.pseudo_aruco_counter = 0
        
        ########### CONNECT TO ROBOT ###########
        if OPERATING_MODE != 2: # not a simulation
            # waits for robot to respond to configure
            count = 0
            Console.info("Connecting to robot")               
            while not self.robot_available:
                self.config_pub.publish(String("Configure"))
                time.sleep(1.0)
                count += 1
            time.sleep(5.0)
        else: # WEBOTS create fake ARUCO logs
            self.sensed_imu_stamp_s = 0 
            self.groundtruth_log = Path("logs/log_" + filename_time + "_pseudo_aruco.csv")
            with self.groundtruth_log.open('w') as f:
                f.write("epoch [s],elapsed [s],x [m],y [m],z [m],roll [deg],pitch [deg],yaw [deg],broadcast\n")

        ########### INITIALISE THRUSTERS ###########
        for i in range(10): #  rad/s
            self.control_pub.publish(Vector3())           
            self.r.sleep()  
            self.initialise_pose = True # Will set to false once the pose is initialised               


        ######## Setup EXIT key if show_laptop not used #####
        if OPERATING_MODE == 0: # robot without show_laptop - stopped via <Ctrl+C> 
            while True:
                try:
                    self.loop()
                except KeyboardInterrupt:
                    Console.info("Ctrl+C pressed. Stopping...")
                    if self.OPERATING_MODE == 0:
                        self.imu_sub.stop()
                        self.sonar_sub.stop()
                        self.lidar_sub.stop()
                    break
                self.r.sleep() 
        ############################## END OF INITIALISATION ##################

    ######## DEFINE FUNCTIONS HERE ##################
    def stopcommand(self):        
        Console.info("Thrusters stopping")
        control_msg = Vector3() # initially 0
        for i in range(10):
            self.control_pub.publish(control_msg)
            self.imu_sub.stop()
            self.r.sleep()
        self.sonar_sub.stop()
        self.lidar_sub.stop()
        Console.info("Thrusters stopped")
        Console.info("Data saved in ",self.filename)
        self.r.sleep()
        
    ######## DEFINE CALLBACKS HERE ##################
    def imu_cb(self, msg: Vector3): 
        self.sensed_imu_yaw_rate_rad_s = msg.z
        self.sensed_imu_stamp_s = time.time()
        self.robot_available = True
        
    def sonar_cb(self,msg: Vector3):
        self.sensed_bottom_depth_m = msg.z/1000        
        self.sensed_bottom_depth_stamp_s = time.time()
        self.robot_available = True

    def command_cb(self,msg: String):
        Console.info(f"Response from robot: {msg.data}")

    # ---------------- LiDAR callback ----------------
    def lidar_callback(self, msg: RBLaserScan):
        if self.sim_init:
            self.sim_time_offset = time.time() - msg.header.stamp
            self.sim_init = False

        self.lidar_timestamp_s = msg.header.stamp + self.sim_time_offset
        self.latest_lidar_received_s = self.lidar_timestamp_s

        ranges = np.array(msg.ranges, dtype=float)
        angles = np.array(msg.angles, dtype=float)

        if len(ranges) != len(angles):
            count = min(len(ranges), len(angles))
            ranges = ranges[:count]
            angles = angles[:count]

        ranges = np.where((ranges > 0.0) & np.isfinite(ranges), ranges, np.nan)
        self.lidar_data_rb = np.column_stack([ranges, angles])

        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        p_eb = Vector(3)
        p_eb[0] = pose[0]
        p_eb[1] = pose[1]
        p_eb[2] = pose[2]

        self.lidar_data = np.full((len(ranges), 2), np.nan)
        z_lm = Vector(2)

        for i, range_m in enumerate(ranges):
            if np.isfinite(range_m):
                z_lm[0] = range_m
                z_lm[1] = angles[i]
                t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

                self.lidar_data[i, 0] = t_em[0]
                self.lidar_data[i, 1] = t_em[1]

        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        self.update_lidar_obstacle_clusters()
        self.update_lidar_sectors()
        self.lidar_new = True
        self.robot_available = True

    # ---------------- LiDAR coordinate transforms ----------------
    def body_point_to_earth(self, point_body):
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        point_body = np.asarray(point_body, dtype=float).reshape(2)

        return np.array([
            pose[0] + c * point_body[0] - s * point_body[1],
            pose[1] + s * point_body[0] + c * point_body[1],
        ])

    # ---------------- LiDAR DBSCAN clustering ----------------
    def clear_lidar_obstacle_clusters(self):
        self.lidar_points_body = np.empty((0, 2))
        self.lidar_cluster_labels = np.array([], dtype=int)
        self.lidar_obstacles = []
        self.lidar_obstacle_centres_body = np.empty((0, 2))
        self.lidar_obstacle_centres_ne = np.empty((0, 2))
        self.lidar_obstacle_distances_m = np.array([])
        self.lidar_obstacle_angles_rad = np.array([])
        self.nearest_lidar_obstacle = None

    def update_lidar_obstacle_clusters(self):
        if self.lidar_data_rb is None:
            self.clear_lidar_obstacle_clusters()
            return

        ranges = self.lidar_data_rb[:, 0]
        angles = self.lidar_data_rb[:, 1]
        valid = np.isfinite(ranges) & np.isfinite(angles)

        if np.count_nonzero(valid) < self.lidar_dbscan_min_points:
            self.clear_lidar_obstacle_clusters()
            return

        valid_ranges = ranges[valid]
        valid_angles = angles[valid] + self.lidar_gamma_bl

        self.lidar_points_body = np.column_stack([
            self.lidar_x_bl + valid_ranges * np.cos(valid_angles),
            self.lidar_y_bl + valid_ranges * np.sin(valid_angles),
        ])

        labels = DBSCAN(
            eps=self.lidar_dbscan_eps_m,
            min_samples=self.lidar_dbscan_min_points,
            n_jobs=1,
        ).fit_predict(self.lidar_points_body)
        self.lidar_cluster_labels = labels

        obstacles = []
        for label in sorted(set(labels)):
            if label == -1:
                continue

            cluster_points = self.lidar_points_body[labels == label]
            centre_body = np.mean(cluster_points, axis=0)
            centre_ne = self.body_point_to_earth(centre_body)
            centre_distance_m = float(np.linalg.norm(centre_body))
            centre_angle_rad = float(np.arctan2(centre_body[1], centre_body[0]))
            min_distance_m = float(np.min(np.linalg.norm(cluster_points, axis=1)))

            obstacles.append({
                "label": int(label),
                "point_count": int(len(cluster_points)),
                "centre_body": centre_body.tolist(),
                "centre_ne": centre_ne.tolist(),
                "distance_m": centre_distance_m,
                "angle_rad": centre_angle_rad,
                "angle_deg": float(np.rad2deg(centre_angle_rad)),
                "min_distance_m": min_distance_m,
            })

        obstacles.sort(key=lambda obstacle: obstacle["distance_m"])
        self.lidar_obstacles = obstacles
        self.nearest_lidar_obstacle = obstacles[0] if obstacles else None

        if obstacles:
            self.lidar_obstacle_centres_body = np.array(
                [obstacle["centre_body"] for obstacle in obstacles],
                dtype=float,
            )
            self.lidar_obstacle_centres_ne = np.array(
                [obstacle["centre_ne"] for obstacle in obstacles],
                dtype=float,
            )
            self.lidar_obstacle_distances_m = np.array(
                [obstacle["distance_m"] for obstacle in obstacles],
                dtype=float,
            )
            self.lidar_obstacle_angles_rad = np.array(
                [obstacle["angle_rad"] for obstacle in obstacles],
                dtype=float,
            )
        else:
            self.lidar_obstacle_centres_body = np.empty((0, 2))
            self.lidar_obstacle_centres_ne = np.empty((0, 2))
            self.lidar_obstacle_distances_m = np.array([])
            self.lidar_obstacle_angles_rad = np.array([])

    # ---------------- LiDAR sector helpers ----------------
    def sector_ranges(self, angle_min, angle_max):
        if self.lidar_data_rb is None:
            return np.array([])

        ranges = self.lidar_data_rb[:, 0]
        angles = self.lidar_data_rb[:, 1]

        mask = (
            (angles > angle_min)
            & (angles < angle_max)
            & np.isfinite(ranges)
        )

        return ranges[mask]

    def sector_min_range(self, angle_min, angle_max):
        vals = self.sector_ranges(angle_min, angle_max)

        if len(vals) == 0:
            return np.inf

        val = np.nanmin(vals)

        if not np.isfinite(val):
            return np.inf

        return val

    def update_lidar_sectors(self):
        self.front_clearance_m = self.sector_min_range(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        self.left_clearance_m = self.sector_min_range(
            self.side_angle_min,
            self.side_angle_max,
        )

        self.right_clearance_m = self.sector_min_range(
            -self.side_angle_max,
            -self.side_angle_min,
        )

        front_ranges = self.sector_ranges(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        if len(front_ranges) == 0:
            front_blocked = False
        else:
            front_blocked = np.sum(front_ranges < self.front_block_threshold) >= self.min_front_close_beams

        left_open = self.left_clearance_m > self.front_block_threshold
        right_open = self.right_clearance_m > self.front_block_threshold

        if front_blocked and left_open and right_open:
            self.map_observation_class = "t_junction_or_end_wall"
        elif front_blocked and left_open:
            self.map_observation_class = "right_angle_left_turn"
        elif front_blocked and right_open:
            self.map_observation_class = "right_angle_right_turn"
        elif front_blocked:
            self.map_observation_class = "blocked_front"
        else:
            self.map_observation_class = "straight_section"

    def front_blocked(self):
        self.update_lidar_sectors()

        front_ranges = self.sector_ranges(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        if len(front_ranges) == 0:
            return False

        close_count = int(np.sum(front_ranges < self.front_block_threshold))
        return close_count >= self.min_front_close_beams

    def groundtruth_callback(self, msg):
        # generate fake aruco data at a set interval
        self.pseudo_aruco_counter += 1 

        t = time.time()
        pose = msg.pose
        n = pose.position.x              
        e = pose.position.y
        d = pose.position.z
        ox = pose.orientation.x
        oy = pose.orientation.y
        oz = pose.orientation.z
        ow = pose.orientation.w
        q = [ox,oy,oz,ow]                
        r = R.from_quat(q)  # note: [x, y, z, w] order
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)  # radians                
        yaw = np.mod(yaw, 360.0)

        if self.pseudo_aruco_counter== 80: 
            self.pseudo_aruco_counter = 0
            self.sensed_pos_stamp_s = t
            self.sensed_pos_northings_m = n
            self.sensed_pos_eastings_m = e
            self.sensed_pos_yaw_rad = np.deg2rad(yaw)
            broadcast = True
        else:
            broadcast = False
                    
        # log groundtruth if running webots simulation
        with self.groundtruth_log.open('a') as f:
            f.write(f"{t},{t-self.starttime},{n},{e},{d},{roll},{pitch},{yaw},{broadcast}\n")
         
    def feedback_control(self, ds, ks = None, kn = None, kg = None):

        if ks == None: ks = 0.1
        if kn == None: kn = 0.1
        if kg == None: kg = 0.1        
        
        dv = ks*ds[0]
        dw = kn*ds[1]+kg*ds[2]
        
        du = Vector(2)
        
        du[0] = dv
        du[1] = dw
        
        return du
    def motion_model(self, state, control_input, dt):
       """
       EKF motion model:
       state x = [N, E, G, Ndot, Edot, Gdot]^T
       control_input = T = [T_R, T_L]^T (thruster forces in N)

       Returns:
           predicted_state (6x1 Vector)
           F               (6x6 Jacobian)
       """

       # Thruster forces in body frame from allocation matrix
       Fb = self.G @ control_input  # [Fx, Fy, tau_z]^T in body frame

       # Convert thrust from body to earth frame
       H_eb = HomogeneousTransformation(state[N:E+1], state[G])
       Fe = H_eb.H_R @ Fb  # [Fx_e, Fy_e, tau_z_e]^T

       # Dynamics in earth frame
       ve = self.robot.model(
           dynamics_translation_e,
           dynamics_rotation_e,
           Fe,
           state[DOTN:DOTG+1],
           dt,
       )  # ve = [Ndot, Edot, Gdot]^T

       # Body-frame velocity
       vb = Inverse(H_eb.H_R) @ ve

       # Twist used for kinematics
       u = Vector(2)
       u[0, 0] = vb[0, 0]  # surge speed v
       u[1, 0] = vb[2, 0]  # yaw rate w

       # Pose update
       p = rigid_body_kinematics(state[N:G+1], u, dt)
       p[2, 0] = p[2, 0] % (2 * np.pi)

       # Build predicted state vector
       predicted_state = Vector(6)
       predicted_state[N]    = p[0]
       predicted_state[E]    = p[1]
       predicted_state[G]    = p[2]
       predicted_state[DOTN] = ve[0]
       predicted_state[DOTE] = ve[1]
       predicted_state[DOTG] = ve[2]

       # Simple Jacobian: integrate velocity (good enough for EKF here)
       F = Identity(6)
       F[N, DOTN]   = dt
       F[E, DOTE]   = dt
       F[G, DOTG]   = dt

       return predicted_state, F

    def empty_measurement(x):
        H = Matrix(5)
        return x, H
    
    ######## MAIN ROBOT LOOP ##################
    def loop(self):
        """This main loop is completed every 0.2 seconds.        
        Once initialised, it repeats until stopped.        
        It runs sequentially so consider how to structure your code.        
        You won't receive data from the IMU or ARUCO in every loop. 
        Don't make the loop rely on new data.
        """
        current_epoch_s = time.time()
        self.timefromstart = current_epoch_s - self.starttime

        ### RECEIVE SENSOR DATA ##############################
        self.sensed_pos_stamp_s = None
        self.sensed_pos_northings_m = None
        self.sensed_pos_eastings_m = None
        self.sensed_pos_yaw_rad = None

        sensed_pos = self.aruco_driver.read()
        if sensed_pos is not None:
            self.sensed_pos_stamp_s = sensed_pos[0]
            self.sensed_pos_northings_m = sensed_pos[1]
            self.sensed_pos_eastings_m = sensed_pos[2]
            self.sensed_pos_yaw_rad = sensed_pos[6]
            print(
                "Received position update from",
                current_epoch_s - self.sensed_pos_stamp_s,
                "seconds ago",
            )

        if self.initialise_pose and self.sensed_pos_northings_m is not None:
            self.mu[N] = self.sensed_pos_northings_m
            self.mu[E] = self.sensed_pos_eastings_m
            self.mu[G] = self.sensed_pos_yaw_rad
            self.mu[DOTN] = 0
            self.mu[DOTE] = 0
            self.mu[DOTG] = 0

            self.p_robot[0] = self.mu[N]
            self.p_robot[1] = self.mu[E]
            self.p_robot[2] = self.mu[G]
            self.v_robot[0] = self.mu[DOTN]
            self.v_robot[1] = self.mu[DOTE]
            self.v_robot[2] = self.mu[DOTG]
            self.integrated_yaw = self.sensed_pos_yaw_rad
            self.initialise_pose = False
            print("Initialised pose")

        imu_fresh = (
            self.sensed_imu_stamp_s is not None
            and current_epoch_s - self.sensed_imu_stamp_s < self.lastdt
        )

        if imu_fresh:
            if self.sensed_imu_prev_stamp_s is not None:
                dt_imu = self.sensed_imu_stamp_s - self.sensed_imu_prev_stamp_s
                if dt_imu <= 0 or dt_imu > 1.0:
                    dt_imu = self.lastdt
            else:
                dt_imu = self.lastdt

            print(
                "Received IMU update from",
                current_epoch_s - self.sensed_imu_stamp_s,
                "seconds ago",
            )

            self.sensed_imu_prev_stamp_s = self.sensed_imu_stamp_s
            self.sensed_yaw_rate = self.sensed_imu_yaw_rate_rad_s
            self.integrated_yaw += self.sensed_yaw_rate * dt_imu
            self.integrated_yaw %= 2 * np.pi

        if (
            self.sensed_bottom_depth_stamp_s is not None
            and current_epoch_s - self.sensed_bottom_depth_stamp_s < self.lastdt
        ):
            print(
                "Received Echosounder update from",
                current_epoch_s - self.sensed_bottom_depth_stamp_s,
                "seconds ago",
            )

        if self.sensed_imu_stamp_s is not None or self.OPERATING_MODE == 2:
            ### EKF PREDICT/UPDATE ##############################
            right_N = rpm2N(-self.right_rate * 60 / (2 * np.pi))
            left_N = rpm2N(self.left_rate * 60 / (2 * np.pi))
            u_thrusters = l2m([right_N, left_N])
            self.mu, self.Sigma = extended_kalman_filter_predict(
                self.mu,
                self.Sigma,
                u_thrusters,
                self.motion_model,
                self.Q,
                self.lastdt,
            )

            if self.sensed_pos_stamp_s is not None:
                z_pose = Vector(6)
                z_pose[N] = self.sensed_pos_northings_m
                z_pose[E] = self.sensed_pos_eastings_m
                z_pose[G] = self.sensed_pos_yaw_rad

                self.mu, self.Sigma = extended_kalman_filter_update(
                    self.mu,
                    self.Sigma,
                    z_pose,
                    h_pose_update,
                    self.R_pose,
                    wrap_index=G,   
                )

            if imu_fresh and self.sensed_yaw_rate is not None:
                z_rate = Vector(6)
                z_rate[DOTG] = self.sensed_yaw_rate

                self.mu, self.Sigma = extended_kalman_filter_update(
                    self.mu,
                    self.Sigma,
                    z_rate,
                    h_grate_update,
                    self.R_grate,
                )

            self.p_robot[0] = self.mu[N]
            self.p_robot[1] = self.mu[E]
            self.p_robot[2] = self.mu[G]
            self.v_robot[0] = self.mu[DOTN]
            self.v_robot[1] = self.mu[DOTE]
            self.v_robot[2] = self.mu[DOTG]
            self.Yaw = self.p_robot[2][0]
            self.North = self.p_robot[0][0]
            self.East = self.p_robot[1][0]

            ### BASIC EKF TRAJECTORY CONTROL ####################
            t = self.timefromstart
            p_ref, u_ref = self.s.p_u_sample(t) 

            dp = p_ref - self.p_robot
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi              
            H_eb = HomogeneousTransformation(self.p_robot[0:2], self.p_robot[2])
            ds = Inverse(H_eb.H_R) @ dp

            if self.kn == None and self.kg == None:
                self.kn = 2 * u_ref[0] / (self.L**2)
                self.kg = u_ref[0] / self.L
                self.u = Vector(2)

            du = self.feedback_control(ds, self.ks, self.kn, self.kg)
            self.u = u_ref + du

            self.u[1, 0] = np.clip(self.u[1, 0], -self.w_max, self.w_max)
            self.u[0, 0] = np.clip(self.u[0, 0], -self.v_max, self.v_max)

            self.kn = 2 * self.u[0] / (self.L**2)
            self.kg = self.u[0] / self.L
            self.prev_sensed = t 
            self.U = self.u.T

            v = self.U[0][0]
            w = self.U[0][1]
            F_x = self.robot.k_drag * v * abs(v)
            tau_z = self.robot.B_66 * w

            thrust = np.linalg.pinv(self.G) @ l2m([F_x, 0, tau_z])
            rpm_R = -N2rpm(thrust[0][0])
            rpm_L = N2rpm(thrust[1][0])

            self.right_rate = float(np.clip(rpm_R * (2 * np.pi / 60), -200, 200))
            self.left_rate = float(np.clip(rpm_L * (2 * np.pi / 60), -200, 200))

            control_msg = Vector3()
            control_msg.x = int(self.right_rate)
            control_msg.y = int(self.left_rate)

            self.control_pub.publish(control_msg)
            print('Prop rates: R=',self.right_rate,', L=',self.left_rate,'rad/s')

            if self.timefromstart >= self.trajectory_duration_s and np.isnan(self.s.t_complete):
                self.s.t_complete = self.timefromstart
            
        ### LOG DATA ##############################
        with self.filename.open("a") as f:
            f.write(f"{current_epoch_s},{self.timefromstart},{self.right_rate},{self.left_rate},{self.lastdt},{self.Yaw},{self.North},{self.East},{self.sensed_yaw_rate},{self.integrated_yaw},{self.sensed_imu_stamp_s},{self.sensed_pos_northings_m},{self.sensed_pos_eastings_m},{self.sensed_pos_yaw_rad}, {self.sensed_pos_stamp_s}, {self.sensed_bottom_depth_stamp_s},{self.sensed_bottom_depth_m}\n")            
        
        ### VISUALISE DATA ##############################
        if self.OPERATING_MODE != 0:
            reference_path = getattr(self.s, "P_arc", getattr(self.s, "P", None))
            mission_complete = not np.isnan(getattr(self.s, "t_complete", np.nan))
            return(self.right_rate, self.left_rate, self.lastdt, self.Yaw, self.North, self.East, self.sensed_yaw_rate,self.integrated_yaw, self.sensed_imu_stamp_s, self.sensed_pos_northings_m, self.sensed_pos_eastings_m, self.sensed_pos_yaw_rad, self.sensed_pos_stamp_s, self.waypoints, reference_path, self.sensed_bottom_depth_m, self.sensed_bottom_depth_stamp_s, mission_complete, self.lidar_data)



        ############################# END MAIN LOOP ###########################
        
def main():
    LaptopController(OPERATING_MODE = 0)
    
if __name__ == "__main__":
    main()
            
