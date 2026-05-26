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

# Keep heading errors continuous for the APF steering law.
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

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
            f.write("EpochTime(s),TimeFromStart(s),right_prop_rate(rad/s),left_prop_rate(rad/s),LastDT(s),Yaw(rad),North(m),East(m),IMUSensedYawRate(rad/s),IMUIntegratedYaw(rad),IMUSensedTimeStamp(s),ARUCOSensedNorth(m),ARUCOSensedEast(m),ARUCOSensedYaw(rad),ArucoSensedTimeStamp(s),DepthTimeStamp(s),Depth(m),APFMode,APFEncounter,APFSide,NearestObstacleNorth(m),NearestObstacleEast(m),NearestObstacleDistance(m),COLREGDCPA(m),COLREGTCPA(s)\n")
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
        self.front_block_threshold = 0.8
        self.min_front_close_beams = 5
        self.side_angle_min = np.deg2rad(45)
        self.side_angle_max = np.deg2rad(120)

        self.map_observation_class = "unknown"
        self.front_clearance_m = np.inf
        self.left_clearance_m = np.inf
        self.right_clearance_m = np.inf

        # ---------------- LiDAR/APF navigation parameters ----------------
        # APF works in the robot body frame but keeps the mission goal in earth N/E.
        self.apf_goal_ne = np.array([goal_north, goal_east], dtype=float)
        # Potential field gains follow the reference APF structure:
        # attractive field + static repulsive field + dynamic repulsive field.
        self.apf_alpha0 = 3.0
        self.apf_beta = 5.0
        self.apf_rho0 = 5.5
        # COLREG reasoning is only applied to tracked moving obstacles with CPA risk.
        # k_m scales the dynamic influence range with relative speed, matching the demo rho0 idea.
        self.apf_colreg_k_m = 1.5
        # CPA must happen within this horizon before the COLREG side force is injected.
        self.apf_colreg_time_horizon_s = 6.0
        # DCPA below this safety domain means the predicted closest approach is unsafe.
        self.apf_colreg_safety_domain_m = 1.0
        # Keep COLREG activation local so distant tracks do not dominate the LiDAR APF.
        self.apf_colreg_rho0_max_m = 4.5
        self.apf_encounter_range_m = self.apf_colreg_rho0_max_m
        # Dynamic obstacles are projected forward as virtual obstacles.
        self.apf_t_pre = 2.0
        self.apf_pred_dt = 0.25
        self.apf_too_close_m = 0.6
        # Control conversion parameters from APF force to body twist [v, w].
        self.apf_gradient_weight = 0.55
        self.apf_heading_gain = 0.8
        self.apf_force_gain = 0.9
        self.apf_deviation_gain = 1.5
        self.apf_side_bias_gain = 0.5
        self.apf_final_goal_ne = np.array([goal_north, goal_east], dtype=float)
        # APF uses a point ahead on the route as its attraction target when avoidance is active.
        self.apf_route_lookahead_s = 5.0
        # APF is blended as an avoidance correction, while route tracking remains the main controller.
        self.apf_avoidance_turn_gain = 0.4
        # Limit the desired APF heading so the USV turns on an arc instead of spinning in place.
        self.apf_heading_change_limit_rad = np.deg2rad(35.0)
        self.apf_turn_min_speed = 0.04
        self.apf_turn_align_rate_rad_s = np.deg2rad(12.0)
        self.apf_max_avoidance_heading_rad = np.deg2rad(85.0)
        self.apf_eps = 1e-3
        self.apf_a_max = 0.4
        self.apf_goal_tolerance_m = 0.30
        # Runtime APF state used for logging, acceleration limiting and plots.
        self.apf_prev_v_cmd = 0.0
        self.apf_force_body = np.zeros(2, dtype=float)
        self.apf_target_body = np.zeros(2, dtype=float)
        self.apf_guidance_mode = "goal"
        self.apf_encounter_mode = "none"
        self.apf_encounter_bearing_deg = np.nan
        self.apf_avoidance_side_sign = 0.0
        # Diagnostics for the selected COLREG target. These values are printed in the loop.
        self.apf_colreg_active = False
        self.apf_colreg_rule = "none"
        self.apf_colreg_dcpa_m = np.nan
        self.apf_colreg_tcpa_s = np.nan
        self.apf_colreg_rho0_m = np.nan
        self.apf_colreg_closing_speed_m_s = np.nan
        # Encounter-sector thresholds. Bearing is measured from own heading, clockwise to starboard.
        self.apf_head_on_half_angle_deg = 22.5
        self.apf_overtaking_half_angle_deg = 67.5
        self.apf_same_heading_limit_deg = 45.0
        self.apf_opposite_heading_limit_deg = 135.0
        self.apf_crossing_a_limit_deg = 112.5
        self.apf_crossing_b_start_deg = 247.5
        # Hold the chosen side so LiDAR noise near a sector boundary does not flip commands.
        self.apf_side_lock_s = 1.5
        self.apf_side_lock_enter_range_m = self.apf_rho0
        self.apf_side_lock_exit_margin_m = 0.75
        self.apf_side_lock_exit_range_m = self.apf_side_lock_enter_range_m + self.apf_side_lock_exit_margin_m
        self.apf_side_lock_front_half_angle_rad = np.deg2rad(100.0)
        self.apf_side_lock_sign = 0.0
        self.apf_side_lock_until_s = 0.0
        self.apf_side_lock_active = False
        self.apf_side_lock_distance_m = np.inf
        # Lightweight track state estimates obstacle velocity from existing DBSCAN clusters.
        self.apf_dynamic_speed_threshold_m_s = 0.03
        self.apf_track_association_m = 0.60
        self.apf_track_timeout_s = 1.0
        self.apf_next_track_id = 1
        self.apf_obstacle_tracks = []
        if self.OPERATING_MODE == 2:
            self.configure_webots_overtaking_apf()
        
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
        v = 0.16 if self.OPERATING_MODE == 2 else 0.1
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
    def configure_webots_overtaking_apf(self):
        # Tuned for the Webots front-obstacle world: a slower vessel starts ahead on
        # the same route, so the own ship should make one smooth pass and then rejoin.
        self.apf_alpha0 = 1.8
        self.apf_beta = 6.0
        self.apf_rho0 = 2.8
        self.apf_colreg_time_horizon_s = 10.0
        self.apf_colreg_safety_domain_m = 0.75
        self.apf_colreg_rho0_max_m = 3.2
        self.apf_encounter_range_m = self.apf_colreg_rho0_max_m
        self.apf_t_pre = 3.0
        self.apf_too_close_m = 0.45
        self.apf_gradient_weight = 0.65
        self.apf_side_bias_gain = 0.25
        self.apf_route_lookahead_s = 18.0
        self.apf_avoidance_turn_gain = 0.6
        self.apf_max_avoidance_heading_rad = np.deg2rad(70.0)
        self.apf_side_lock_s = 5.0
        self.apf_side_lock_enter_range_m = self.apf_rho0
        self.apf_side_lock_exit_margin_m = 0.50
        self.apf_side_lock_exit_range_m = (
            self.apf_side_lock_enter_range_m + self.apf_side_lock_exit_margin_m
        )
        self.apf_track_association_m = 0.80
        self.apf_track_timeout_s = 1.5

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
        # APF reuses DBSCAN clusters to estimate obstacle motion; raw LiDAR handling is unchanged.
        self.update_apf_obstacle_tracks(self.lidar_timestamp_s)
        self.update_lidar_sectors()
        self.lidar_new = True
        self.robot_available = True

    # ---------------- LiDAR coordinate transforms ----------------
    def earth_vector_to_body(self, vector_ne):
        # Body frame convention: x is forward, y is left, gamma is yaw in earth frame.
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        vector_ne = np.asarray(vector_ne, dtype=float).reshape(2)

        return np.array([
            c * vector_ne[0] + s * vector_ne[1],
            -s * vector_ne[0] + c * vector_ne[1],
        ])

    def earth_point_to_body(self, point_ne):
        # Point transform is a vector transform after subtracting the EKF robot position.
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        return self.earth_vector_to_body(np.asarray(point_ne, dtype=float).reshape(2) - pose[0:2])

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

    # ---------------- APF encounter-sector helpers ----------------
    def apf_relative_bearing_deg(self, body_angle_rad):
        # Body LiDAR angles are positive to port/left; the encounter diagram is clockwise.
        return float((-np.rad2deg(wrap_angle(body_angle_rad))) % 360.0)

    def apf_signed_starboard_bearing_deg(self, body_angle_rad):
        # Signed bearing used by COLREG logic: positive is starboard/right, negative is port/left.
        return float(np.rad2deg(wrap_angle(-body_angle_rad)))

    def apf_encounter_situation(self, relative_bearing_deg):
        # Bearing-only fallback. Full COLREG classification also checks relative velocity and DCPA.
        bearing_deg = float(relative_bearing_deg) % 360.0

        if (
            bearing_deg <= self.apf_head_on_half_angle_deg
            or bearing_deg >= 360.0 - self.apf_head_on_half_angle_deg
        ):
            return "head_on"

        if bearing_deg <= self.apf_crossing_a_limit_deg:
            return "crossing_from_starboard"

        if bearing_deg >= self.apf_crossing_b_start_deg:
            return "crossing_from_port"

        return "other"

    def apf_requested_side_from_encounter(self, encounter_mode, body_angle_rad):
        # Positive side sign means bias APF to port/left; negative means starboard/right.
        # Starboard-side manoeuvres therefore use -1.0 in this body-frame convention.
        if encounter_mode in ("head_on", "crossing_a", "crossing_from_starboard"):
            return -1.0

        if encounter_mode in ("crossing_b", "crossing_from_port", "overtaking"):
            return 1.0

        return 0.0

    def apf_requested_side_from_obstacle_angle(self, body_angle_rad):
        # For non-COLREG obstacles, lock the pass side from obstacle bearing.
        # Positive LiDAR/body angle is port/left, so bias to starboard/right.
        body_angle_rad = float(wrap_angle(body_angle_rad))

        if not np.isfinite(body_angle_rad):
            return 0.0

        if abs(body_angle_rad) > self.apf_side_lock_front_half_angle_rad:
            return 0.0

        centre_deadband_rad = np.deg2rad(5.0)
        if abs(body_angle_rad) <= centre_deadband_rad:
            if self.left_clearance_m > self.right_clearance_m + 0.05:
                return 1.0
            if self.right_clearance_m > self.left_clearance_m + 0.05:
                return -1.0
            return 1.0

        return -1.0 if body_angle_rad > 0.0 else 1.0

    def apf_own_velocity_body(self):
        # Use EKF velocity for CPA. If the EKF velocity is still near zero after startup,
        # fall back to the previous forward command so collision checks do not go blind.
        vel_ne = np.asarray(self.v_robot[0:2], dtype=float).reshape(2)
        vel_body = self.earth_vector_to_body(vel_ne)

        if not np.isfinite(vel_body).all():
            vel_body = np.zeros(2, dtype=float)

        measured_speed = float(np.linalg.norm(vel_body))
        if measured_speed < self.apf_dynamic_speed_threshold_m_s and self.apf_prev_v_cmd > measured_speed:
            vel_body = np.array([self.apf_prev_v_cmd, 0.0], dtype=float)

        return vel_body

    def apf_cpa_metrics(self, obs_pos_body, obs_vel_body, own_vel_body):
        # CPA is evaluated in the own-ship body frame:
        # TCPA is time to closest approach, DCPA is the distance at that instant.
        obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        own_vel_body = np.asarray(own_vel_body, dtype=float).reshape(2)

        rel_vel = obs_vel_body - own_vel_body
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))

        if rel_speed_sq < 1e-9:
            return np.inf, float(np.linalg.norm(obs_pos_body))

        tcpa = -float(np.dot(obs_pos_body, rel_vel)) / rel_speed_sq
        tcpa = max(tcpa, 0.0)
        dcpa = float(np.linalg.norm(obs_pos_body + rel_vel * tcpa))
        return tcpa, dcpa

    def apf_track_for_obstacle(self, obstacle):
        # Match the current DBSCAN cluster back to its velocity track.
        # The track gives us obstacle velocity, while the cluster gives current LiDAR geometry.
        if not self.apf_obstacle_tracks:
            return None

        centre_ne = np.asarray(obstacle.get("centre_ne", [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(centre_ne).all():
            return None

        now = float(self.latest_lidar_received_s if self.latest_lidar_received_s is not None else time.time())
        best_track = None
        best_distance = np.inf

        for track in self.apf_obstacle_tracks:
            if now - track["stamp_s"] > self.apf_track_timeout_s:
                continue

            dt = max(now - track["stamp_s"], 0.0)
            predicted_pos = track["pos_ne"] + track["vel_ne"] * dt
            distance = float(np.linalg.norm(centre_ne - predicted_pos))

            if distance < best_distance:
                best_distance = distance
                best_track = track

        association_limit = max(self.apf_track_association_m * 1.5, self.lidar_dbscan_eps_m * 2.0)
        if best_distance <= association_limit:
            return best_track

        return None

    def apf_classify_colreg_encounter(self, bearing_signed_deg, obs_heading_body_rad, own_speed, obs_speed):
        # Simplified COLREG classifier adapted to local LiDAR tracks.
        # bearing_signed_deg: positive starboard, negative port.
        # obs_heading_body_rad: obstacle course expressed relative to own heading.
        bearing_signed_deg = float(bearing_signed_deg)
        own_speed = float(own_speed)
        obs_speed = float(obs_speed)

        relative_heading_deg = np.nan
        if obs_speed >= self.apf_dynamic_speed_threshold_m_s:
            relative_heading_deg = abs(float(np.rad2deg(wrap_angle(obs_heading_body_rad))))

        if (
            np.isfinite(relative_heading_deg)
            and abs(bearing_signed_deg) <= self.apf_head_on_half_angle_deg
            and relative_heading_deg > self.apf_opposite_heading_limit_deg
        ):
            # Rule 14: both vessels alter course to starboard in a head-on situation.
            return "head_on", -1.0, "Rule 14 head-on: alter to starboard"

        if (
            np.isfinite(relative_heading_deg)
            and abs(bearing_signed_deg) <= self.apf_overtaking_half_angle_deg
            and relative_heading_deg < self.apf_same_heading_limit_deg
            and own_speed > obs_speed + self.apf_dynamic_speed_threshold_m_s
        ):
            # Rule 13: overtaking vessel keeps clear. This implementation prefers port.
            return "overtaking", 1.0, "Rule 13 overtaking: keep clear, prefer port"

        if 0.0 < bearing_signed_deg <= self.apf_crossing_a_limit_deg:
            # Rule 15: target on starboard side, own ship gives way by turning starboard.
            return "crossing_from_starboard", -1.0, "Rule 15 give-way: alter to starboard"

        if -self.apf_crossing_a_limit_deg <= bearing_signed_deg < 0.0:
            # Rule 17: target on port side, own ship is stand-on unless CPA becomes unsafe.
            return "crossing_from_port", 1.0, "Rule 17 stand-on: evasive port only if CPA unsafe"

        return "other", 0.0, "none"

    def apf_colreg_status_for_obstacle(self, obstacle):
        # Build one self-contained risk record for a LiDAR obstacle.
        # The default is inactive so static walls and untracked clusters still use normal APF.
        body_angle_rad = float(obstacle.get("angle_rad", 0.0))
        distance_m = float(obstacle.get("min_distance_m", obstacle.get("distance_m", np.inf)))
        status = {
            "active": False,
            "side_sign": 0.0,
            "encounter": "static_obstacle",
            "rule": "none",
            "bearing_deg": self.apf_relative_bearing_deg(body_angle_rad),
            "body_angle_rad": body_angle_rad,
            "distance_m": distance_m,
            "dcpa_m": np.nan,
            "tcpa_s": np.nan,
            "rho0_m": np.nan,
            "closing_speed_m_s": np.nan,
        }

        track = self.apf_track_for_obstacle(obstacle)
        if track is None or track.get("hit_count", 0) < 2:
            # A single detection has no reliable velocity, so it cannot be a COLREG target yet.
            return status

        # Predict the matched track to the current scan time before calculating bearings and CPA.
        now = float(self.latest_lidar_received_s if self.latest_lidar_received_s is not None else time.time())
        dt = max(now - track["stamp_s"], 0.0)
        obs_pos_ne = track["pos_ne"] + track["vel_ne"] * dt
        obs_pos_body = self.earth_point_to_body(obs_pos_ne)
        obs_vel_body = self.earth_vector_to_body(track["vel_ne"])

        if not np.isfinite(obs_pos_body).all() or not np.isfinite(obs_vel_body).all():
            return status

        obs_speed = float(np.linalg.norm(obs_vel_body))
        if obs_speed < self.apf_dynamic_speed_threshold_m_s:
            # Slow or static objects are handled by the static LiDAR potential field.
            return status

        own_vel_body = self.apf_own_velocity_body()
        own_speed = float(np.linalg.norm(own_vel_body))
        distance_m = float(np.linalg.norm(obs_pos_body))

        if distance_m < 1e-6:
            return status

        # Recompute bearing from the predicted track position, not just the cluster centroid.
        body_angle_rad = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
        bearing_signed_deg = self.apf_signed_starboard_bearing_deg(body_angle_rad)
        obs_heading_body_rad = float(np.arctan2(obs_vel_body[1], obs_vel_body[0]))
        encounter, requested_side, rule = self.apf_classify_colreg_encounter(
            bearing_signed_deg,
            obs_heading_body_rad,
            own_speed,
            obs_speed,
        )

        tcpa_s, dcpa_m = self.apf_cpa_metrics(obs_pos_body, obs_vel_body, own_vel_body)
        rel_speed = float(np.linalg.norm(own_vel_body - obs_vel_body))
        closing_speed = float(np.dot(own_vel_body - obs_vel_body, obs_pos_body / distance_m))
        if (
            abs(bearing_signed_deg) <= self.apf_overtaking_half_angle_deg
            and own_speed > obs_speed + self.apf_dynamic_speed_threshold_m_s
            and 0.0 < closing_speed <= max(0.08, 0.6 * own_speed)
        ):
            # In the Webots overtaking setup the obstacle is ahead and moving in the
            # same direction. LiDAR-centroid velocity can briefly look like head-on
            # while the own ship is turning, so closing speed is used as a stabiliser.
            encounter = "overtaking"
            requested_side = 1.0
            rule = "Rule 13 overtaking: keep clear, prefer port"
        # Dynamic influence range from the demo: faster relative motion expands rho0.
        rho0_m = self.apf_colreg_k_m * np.sqrt(
            (rel_speed * self.apf_colreg_time_horizon_s) ** 2
            + self.apf_colreg_safety_domain_m ** 2
        )
        rho0_m = float(np.clip(rho0_m, self.apf_rho0, self.apf_colreg_rho0_max_m))

        active = (
            requested_side != 0.0
            and distance_m <= rho0_m
            and closing_speed > 0.0
            and dcpa_m < self.apf_colreg_safety_domain_m
            and tcpa_s <= self.apf_colreg_time_horizon_s
        )

        if distance_m <= self.apf_too_close_m and requested_side != 0.0:
            # Emergency override: if the moving vessel is already very close, enforce the rule side.
            active = True

        status.update({
            "active": bool(active),
            "side_sign": requested_side if active else 0.0,
            "encounter": encounter,
            "rule": rule,
            "bearing_deg": self.apf_relative_bearing_deg(body_angle_rad),
            "body_angle_rad": body_angle_rad,
            "distance_m": distance_m,
            "dcpa_m": dcpa_m,
            "tcpa_s": tcpa_s,
            "rho0_m": rho0_m,
            "closing_speed_m_s": closing_speed,
        })
        return status

    def apf_primary_encounter_obstacle(self):
        # Choose the nearest LiDAR cluster inside the APF influence range for rule reasoning.
        if not self.lidar_obstacles:
            return None

        candidates = []
        for obstacle in self.lidar_obstacles:
            distance_m = float(obstacle.get("min_distance_m", obstacle.get("distance_m", np.inf)))
            if np.isfinite(distance_m) and distance_m <= self.apf_encounter_range_m:
                candidates.append(obstacle)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda obstacle: float(obstacle.get("min_distance_m", obstacle.get("distance_m", np.inf))),
        )

    def apf_lock_avoidance_side(
        self,
        requested_side,
        encounter_mode,
        obstacle_distance_m=None,
        safe_distance_m=None,
    ):
        # Keep the side decision while the obstacle remains inside the safe exit range.
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        safe_distance_m = (
            self.apf_side_lock_exit_range_m
            if safe_distance_m is None
            else float(safe_distance_m)
        )
        distance_m = np.nan if obstacle_distance_m is None else float(obstacle_distance_m)
        has_distance = np.isfinite(distance_m)
        obstacle_still_close = has_distance and distance_m <= safe_distance_m
        self.apf_side_lock_distance_m = distance_m if has_distance else np.inf

        if requested_side == 0.0:
            if self.apf_side_lock_sign != 0.0:
                if obstacle_still_close:
                    self.apf_side_lock_active = True
                    self.apf_side_lock_until_s = max(
                        self.apf_side_lock_until_s,
                        now_s + self.apf_side_lock_s,
                    )
                    return self.apf_side_lock_sign

                if not has_distance and now_s < self.apf_side_lock_until_s:
                    self.apf_side_lock_active = True
                    return self.apf_side_lock_sign

            self.apf_side_lock_sign = 0.0
            self.apf_side_lock_until_s = now_s
            self.apf_side_lock_active = False
            return 0.0

        if (
            self.apf_side_lock_sign != 0.0
            and (obstacle_still_close or now_s < self.apf_side_lock_until_s)
        ):
            self.apf_side_lock_active = True
            if obstacle_still_close:
                self.apf_side_lock_until_s = max(
                    self.apf_side_lock_until_s,
                    now_s + self.apf_side_lock_s,
                )
            return self.apf_side_lock_sign

        self.apf_side_lock_sign = float(np.sign(requested_side))
        self.apf_side_lock_until_s = now_s + self.apf_side_lock_s
        self.apf_side_lock_active = True
        return self.apf_side_lock_sign

    def apf_encounter_avoidance_side(self):
        # Apply COLREG side selection only when a tracked moving obstacle has unsafe CPA.
        if not self.lidar_obstacles:
            self.apf_encounter_mode = "none"
            self.apf_colreg_rule = "none"
            self.apf_encounter_bearing_deg = np.nan
            self.apf_colreg_active = False
            self.apf_colreg_dcpa_m = np.nan
            self.apf_colreg_tcpa_s = np.nan
            self.apf_colreg_rho0_m = np.nan
            self.apf_colreg_closing_speed_m_s = np.nan
            self.apf_avoidance_side_sign = self.apf_lock_avoidance_side(0.0, "none")
            return self.apf_avoidance_side_sign

        statuses = []
        for obstacle in self.lidar_obstacles:
            distance_m = float(obstacle.get("min_distance_m", obstacle.get("distance_m", np.inf)))
            if np.isfinite(distance_m) and distance_m <= self.apf_encounter_range_m:
                statuses.append(self.apf_colreg_status_for_obstacle(obstacle))

        if not statuses:
            self.apf_encounter_mode = "none"
            self.apf_colreg_rule = "none"
            self.apf_encounter_bearing_deg = np.nan
            self.apf_colreg_active = False
            self.apf_colreg_dcpa_m = np.nan
            self.apf_colreg_tcpa_s = np.nan
            self.apf_colreg_rho0_m = np.nan
            self.apf_colreg_closing_speed_m_s = np.nan
            self.apf_avoidance_side_sign = self.apf_lock_avoidance_side(0.0, "none", np.inf)
            return self.apf_avoidance_side_sign

        active_statuses = [status for status in statuses if status["active"]]
        if active_statuses:
            # If more than one vessel is risky, prioritise the smallest DCPA, then earliest TCPA.
            selected = min(
                active_statuses,
                key=lambda status: (
                    status["dcpa_m"] if np.isfinite(status["dcpa_m"]) else np.inf,
                    status["tcpa_s"] if np.isfinite(status["tcpa_s"]) else np.inf,
                    status["distance_m"],
                ),
            )
            requested_side = selected["side_sign"]
            if np.isfinite(selected["rho0_m"]):
                safe_distance_m = selected["rho0_m"] + self.apf_side_lock_exit_margin_m
            else:
                safe_distance_m = self.apf_side_lock_exit_range_m
        else:
            # Keep diagnostics from the nearest candidate and lock a pass side once it is close enough.
            selected = min(statuses, key=lambda status: status["distance_m"])
            requested_side = 0.0
            safe_distance_m = self.apf_side_lock_exit_range_m

            if selected["distance_m"] <= self.apf_side_lock_enter_range_m:
                requested_side = self.apf_requested_side_from_obstacle_angle(
                    selected.get("body_angle_rad", 0.0),
                )

        selected_body_angle = float(selected.get("body_angle_rad", np.nan))
        if (
            np.isfinite(selected_body_angle)
            and abs(wrap_angle(selected_body_angle)) > self.apf_side_lock_front_half_angle_rad
        ):
            # Once the obstacle is well abeam or behind, stop biasing to the avoidance side
            # so the route controller can bring the USV back to the original voyage line.
            requested_side = 0.0
            safe_distance_m = 0.0

        side_sign = self.apf_lock_avoidance_side(
            requested_side,
            selected["encounter"],
            selected["distance_m"],
            safe_distance_m,
        )

        self.apf_encounter_mode = selected["encounter"]
        self.apf_colreg_rule = selected["rule"]
        self.apf_encounter_bearing_deg = selected["bearing_deg"]
        self.apf_colreg_active = bool(selected["active"])
        self.apf_colreg_dcpa_m = selected["dcpa_m"]
        self.apf_colreg_tcpa_s = selected["tcpa_s"]
        self.apf_colreg_rho0_m = selected["rho0_m"]
        self.apf_colreg_closing_speed_m_s = selected["closing_speed_m_s"]
        self.apf_avoidance_side_sign = side_sign
        return side_sign

    # ---------------- APF obstacle tracking and potentials ----------------
    def update_apf_obstacle_tracks(self, stamp_s):
        # Associate current DBSCAN centroids with previous centroids to estimate obstacle velocity.
        # This adds dynamic APF data without changing the LiDAR or clustering pipeline.
        now = float(stamp_s if stamp_s is not None else time.time())
        detections = []

        for obstacle in self.lidar_obstacles:
            centre_ne = np.asarray(obstacle["centre_ne"], dtype=float).reshape(2)
            if np.isfinite(centre_ne).all():
                detections.append(centre_ne)

        if not detections:
            # Keep recent tracks briefly so one missed scan does not reset velocity estimates.
            for track in self.apf_obstacle_tracks:
                track["miss_count"] += 1

            self.apf_obstacle_tracks = [
                track for track in self.apf_obstacle_tracks
                if now - track["stamp_s"] <= self.apf_track_timeout_s
                and track["miss_count"] <= 5
            ]
            return

        detections = np.asarray(detections, dtype=float)
        candidates = []

        for track_index, track in enumerate(self.apf_obstacle_tracks):
            # Predict each track to the current scan time before nearest-neighbour matching.
            dt = max(now - track["stamp_s"], 0.0)
            predicted_pos = track["pos_ne"] + track["vel_ne"] * dt

            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(detection - predicted_pos))
                candidates.append((distance, track_index, detection_index))

        candidates.sort(key=lambda item: item[0])
        assigned_tracks = set()
        assigned_detections = set()

        for distance, track_index, detection_index in candidates:
            if distance > self.apf_track_association_m:
                break
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue

            track = self.apf_obstacle_tracks[track_index]
            detection = detections[detection_index]
            dt = now - track["stamp_s"]

            if 1e-3 < dt <= self.apf_track_timeout_s:
                measured_vel = (detection - track["pos_ne"]) / dt
                if track["hit_count"] <= 1:
                    track["vel_ne"] = measured_vel
                else:
                    # Smooth velocity to reduce APF jitter from small centroid shifts.
                    track["vel_ne"] = 0.5 * track["vel_ne"] + 0.5 * measured_vel
            else:
                track["vel_ne"] = np.zeros(2, dtype=float)

            track["pos_ne"] = detection
            track["stamp_s"] = now
            track["hit_count"] += 1
            track["miss_count"] = 0
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)

        for track_index, track in enumerate(self.apf_obstacle_tracks):
            if track_index not in assigned_tracks:
                track["miss_count"] += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue

            # New centroid starts as a static track until a second observation provides velocity.
            self.apf_obstacle_tracks.append({
                "id": self.apf_next_track_id,
                "pos_ne": detection,
                "vel_ne": np.zeros(2, dtype=float),
                "stamp_s": now,
                "hit_count": 1,
                "miss_count": 0,
            })
            self.apf_next_track_id += 1

        # Drop stale tracks so old obstacles do not keep generating virtual obstacles.
        self.apf_obstacle_tracks = [
            track for track in self.apf_obstacle_tracks
            if now - track["stamp_s"] <= self.apf_track_timeout_s
            and track["miss_count"] <= 5
        ]

    def apf_dynamic_obstacles_body(self):
        # Only tracks with enough speed are treated as dynamic; slow clusters remain static APF points.
        now = float(self.latest_lidar_received_s if self.latest_lidar_received_s is not None else time.time())
        dynamic_obstacles = []

        for track in self.apf_obstacle_tracks:
            if now - track["stamp_s"] > self.apf_track_timeout_s:
                continue
            if track["hit_count"] < 2:
                continue

            speed = float(np.linalg.norm(track["vel_ne"]))
            if speed < self.apf_dynamic_speed_threshold_m_s:
                continue

            # Dynamic repulsion is generated in the robot local frame.
            pos_body = self.earth_point_to_body(track["pos_ne"])
            vel_body = self.earth_vector_to_body(track["vel_ne"])

            if np.isfinite(pos_body).all() and np.isfinite(vel_body).all():
                dynamic_obstacles.append((pos_body, vel_body))

        return dynamic_obstacles

    def apf_static_repulsive_potential(self, q, lidar_points_body):
        # Static APF is evaluated directly from body-frame LiDAR points.
        # Max fusion follows the paper/reference code and avoids over-penalising dense point clouds.
        if lidar_points_body is None or len(lidar_points_body) == 0:
            return 0.0

        q = np.asarray(q, dtype=float).reshape(2)
        lidar_points_body = np.asarray(lidar_points_body, dtype=float)

        if lidar_points_body.ndim != 2 or lidar_points_body.shape[1] != 2:
            return 0.0

        diff = q[None, :] - lidar_points_body
        distances = np.linalg.norm(diff, axis=1)
        distances = np.maximum(distances, 1e-3)

        mask = distances < self.apf_rho0
        if not np.any(mask):
            return 0.0

        potential = self.apf_alpha0 * (1.0 / distances[mask] - 1.0 / self.apf_rho0) ** 2
        return float(np.max(potential))

    def apf_dynamic_obstacle_sequences(self, dynamic_obstacles_body):
        # Generate the real obstacle plus future virtual obstacles along its estimated velocity.
        # Later virtual obstacles get larger gain, matching the growing predicted collision risk.
        sequences = []
        imminent_risk = False
        n_step = max(1, int(self.apf_t_pre / self.apf_pred_dt))

        for obs_pos_body, obs_vel_body in dynamic_obstacles_body:
            obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
            obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
            speed = float(np.linalg.norm(obs_vel_body))

            for i in range(n_step + 1):
                q_i = obs_pos_body + obs_vel_body * self.apf_pred_dt * i
                distance = float(np.linalg.norm(q_i))

                if distance < self.apf_too_close_m:
                    # Too-close predicted obstacles switch control to active avoidance.
                    imminent_risk = True
                    break

                if distance < 1e-6:
                    continue

                f1 = max(1.0 - 0.3 * speed, 0.1)
                growth = 1.0 + 0.7 * i * i
                # Obstacles predicted behind the USV are weakened, as in the reasoning APF.
                theta_vo = np.arccos(np.clip(q_i[0] / distance, -1.0, 1.0))

                if theta_vo > np.pi / 2:
                    f2 = max((np.pi - theta_vo) / (np.pi / 2), 0.0) ** 5
                else:
                    f2 = 1.0

                gain = self.apf_alpha0 * f1 * growth * f2
                sequences.append((q_i, gain))

        return sequences, imminent_risk

    def apf_dynamic_repulsive_potential(self, q, dynamic_sequences):
        # Dynamic sequences are also max-fused so a cluster of virtual points does not explode force.
        if not dynamic_sequences:
            return 0.0

        q = np.asarray(q, dtype=float).reshape(2)
        potentials = []

        for obstacle_pos_body, gain in dynamic_sequences:
            distance = float(np.linalg.norm(q - obstacle_pos_body))
            distance = max(distance, 1e-3)

            if distance < self.apf_rho0:
                potentials.append(gain * (1.0 / distance - 1.0 / self.apf_rho0) ** 2)

        if not potentials:
            return 0.0

        return float(np.max(potentials))

    def apf_total_potential(self, q, target_body, lidar_points_body, dynamic_sequences):
        # Comprehensive local APF: target attraction + static LiDAR repulsion + dynamic prediction.
        q = np.asarray(q, dtype=float).reshape(2)
        target_body = np.asarray(target_body, dtype=float).reshape(2)
        attractive = self.apf_beta * float(np.linalg.norm(target_body - q))
        static_repulsive = self.apf_static_repulsive_potential(q, lidar_points_body)
        dynamic_repulsive = self.apf_dynamic_repulsive_potential(q, dynamic_sequences)
        return attractive + static_repulsive + dynamic_repulsive

    def apf_numerical_gradient(self, func, q):
        # Central-difference gradient keeps the APF implementation independent of potential details.
        q = np.asarray(q, dtype=float).reshape(2)
        dx = np.array([self.apf_eps, 0.0])
        dy = np.array([0.0, self.apf_eps])

        dUdx = (func(q + dx) - func(q - dx)) / (2 * self.apf_eps)
        dUdy = (func(q + dy) - func(q - dy)) / (2 * self.apf_eps)

        return np.array([dUdx, dUdy], dtype=float)

    def current_body_surge_speed(self):
        # EKF velocity is stored in earth frame; APF look-ahead needs body-frame surge.
        H_eb = HomogeneousTransformation(self.p_robot[0:2], self.p_robot[2])
        v_body = Inverse(H_eb.H_R) @ self.v_robot
        return float(v_body[0, 0])

    def compute_route_tracking_control(self, t):
        # The route from the start waypoint to the goal waypoint is the primary plan.
        p_ref, u_ref = self.s.p_u_sample(t)

        # Convert route error into the robot body frame:
        # forward error adjusts speed, lateral error adjusts yaw rate.
        current_ne = np.array([self.North, self.East], dtype=float)
        ref_ne = np.array([float(p_ref[0, 0]), float(p_ref[1, 0])], dtype=float)
        error_body = self.earth_vector_to_body(ref_ne - current_ne)
        heading_error = wrap_angle(float(p_ref[2, 0]) - float(self.Yaw))

        ds = Vector(3)
        ds[0, 0] = error_body[0]
        ds[1, 0] = error_body[1]
        ds[2, 0] = heading_error

        # Feed-forward route velocity plus feedback keeps the robot on the planned route.
        u_track = u_ref + self.feedback_control(ds, self.ks, self.kn, self.kg)
        return p_ref, u_ref, u_track

    def apf_avoidance_needed(self):
        # Keep APF out of route tracking unless an obstacle is close enough to matter.
        if self.front_blocked() or self.apf_side_lock_active:
            return True

        if self.nearest_lidar_obstacle is not None:
            nearest_distance = float(
                self.nearest_lidar_obstacle.get(
                    "min_distance_m",
                    self.nearest_lidar_obstacle.get("distance_m", np.inf),
                )
            )
            nearest_angle = float(self.nearest_lidar_obstacle.get("angle_rad", np.inf))
            obstacle_in_forward_sector = (
                np.isfinite(nearest_angle)
                and abs(wrap_angle(nearest_angle)) <= self.apf_side_lock_front_half_angle_rad
            )
            if (
                np.isfinite(nearest_distance)
                and nearest_distance <= self.apf_rho0
                and obstacle_in_forward_sector
            ):
                return True

        for obs_pos_body, _ in self.apf_dynamic_obstacles_body():
            obs_distance = float(np.linalg.norm(obs_pos_body))
            obs_angle = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
            if (
                obs_distance <= self.apf_rho0
                and abs(wrap_angle(obs_angle)) <= self.apf_side_lock_front_half_angle_rad
            ):
                return True

        return False

    def compute_apf_control(self):
        # Build the APF in the EKF-centred body frame and return body twist [v, w].
        target_body = self.earth_point_to_body(self.apf_goal_ne)
        self.apf_target_body = target_body
        goal_distance = float(np.linalg.norm(target_body))

        u_cmd = Vector(2)
        if goal_distance <= self.apf_goal_tolerance_m:
            # Goal reached: publish zero twist and let the existing mission-complete flag update.
            self.apf_prev_v_cmd = 0.0
            self.apf_force_body = np.zeros(2, dtype=float)
            self.apf_guidance_mode = "arrived"
            self.apf_colreg_active = False
            self.apf_avoidance_side_sign = 0.0
            self.apf_side_lock_sign = 0.0
            self.apf_side_lock_until_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
            self.apf_side_lock_active = False
            return u_cmd

        # Static obstacles come from existing LiDAR points; dynamic obstacles come from cluster tracks.
        lidar_points_body = self.lidar_points_body
        dynamic_obstacles_body = self.apf_dynamic_obstacles_body()
        dynamic_sequences, imminent_risk = self.apf_dynamic_obstacle_sequences(dynamic_obstacles_body)

        def potential(q):
            return self.apf_total_potential(q, target_body, lidar_points_body, dynamic_sequences)

        q0 = np.zeros(2, dtype=float)
        surge_speed = max(self.current_body_surge_speed(), 0.0)
        # Nesterov-style look-ahead samples the field where the USV will soon be.
        q_exceed = np.array([surge_speed * 0.5, 0.0], dtype=float)

        gradient_weight = self.apf_gradient_weight
        if goal_distance < float(np.linalg.norm(q_exceed)):
            # Near the goal, avoid looking past the target.
            gradient_weight = 1.0

        # The virtual APF force is the negative gradient at the current and look-ahead positions.
        grad_now = self.apf_numerical_gradient(potential, q0)
        grad_future = self.apf_numerical_gradient(potential, q_exceed)
        force_body = -gradient_weight * grad_now - (1.0 - gradient_weight) * grad_future

        force_norm = float(np.linalg.norm(force_body))
        if not np.isfinite(force_norm) or force_norm < 1e-6:
            # Degenerate field fallback: keep a tiny force towards the goal.
            force_body = target_body / max(goal_distance, 1e-6) * 1e-3
            force_norm = float(np.linalg.norm(force_body))

        force_offset = wrap_angle(float(np.arctan2(force_body[1], force_body[0])))
        blocked_front = self.front_blocked()
        encounter_side_sign = self.apf_encounter_avoidance_side()

        if encounter_side_sign != 0.0 and (self.apf_colreg_active or self.apf_side_lock_active):
            # Once avoidance starts, keep biasing the selected side until the obstacle exits safely.
            side_bias = self.apf_side_bias_gain * max(force_norm, self.apf_beta)
            force_body = force_body + np.array([0.0, encounter_side_sign * side_bias], dtype=float)
            force_norm = float(np.linalg.norm(force_body))
        elif blocked_front and abs(force_offset) > np.pi / 2:
            # If APF points backward and no encounter rule is active, bias toward the clearer side.
            requested_side = 1.0 if self.left_clearance_m >= self.right_clearance_m else -1.0
            side_sign = self.apf_lock_avoidance_side(
                requested_side,
                "clearance",
                self.front_clearance_m,
            )
            self.apf_avoidance_side_sign = side_sign
            side_bias = self.apf_side_bias_gain * max(force_norm, self.apf_beta)
            force_body = force_body + np.array([0.0, side_sign * side_bias], dtype=float)
            force_norm = float(np.linalg.norm(force_body))

        force_offset = wrap_angle(float(np.arctan2(force_body[1], force_body[0])))
        avoidance_active = (
            blocked_front
            or imminent_risk
            or self.apf_colreg_active
            or self.apf_side_lock_active
        )

        if avoidance_active and abs(force_offset) > self.apf_max_avoidance_heading_rad:
            # Do not let obstacle repulsion turn the USV back along its incoming path.
            side_sign = self.apf_avoidance_side_sign

            if side_sign == 0.0:
                side_sign = np.sign(force_offset)

            if side_sign == 0.0:
                side_sign = 1.0 if self.left_clearance_m >= self.right_clearance_m else -1.0

            capped_offset = side_sign * self.apf_max_avoidance_heading_rad
            force_body = force_norm * np.array(
                [np.cos(capped_offset), np.sin(capped_offset)],
                dtype=float,
            )
            force_offset = capped_offset

        if abs(force_offset) > self.apf_heading_change_limit_rad:
            turn_sign = np.sign(force_offset)

            if turn_sign == 0.0:
                turn_sign = 1.0 if self.left_clearance_m >= self.right_clearance_m else -1.0

            force_offset = turn_sign * self.apf_heading_change_limit_rad
            force_body = force_norm * np.array(
                [np.cos(force_offset), np.sin(force_offset)],
                dtype=float,
            )

        self.apf_force_body = force_body
        target_offset = wrap_angle(float(np.arctan2(target_body[1], target_body[0])))
        avoidance_risk = imminent_risk or self.apf_colreg_active or self.apf_side_lock_active

        # In this body-frame convention, positive offset means the desired force is to port/left.
        w_limit = self.w_max
        if not avoidance_risk and abs(target_offset) > np.pi / 2:
            w_limit = min(w_limit, self.apf_turn_align_rate_rad_s)

        w_cmd = self.apf_heading_gain * force_offset
        w_cmd = float(np.clip(w_cmd, -w_limit, w_limit))

        # Linear speed uses force magnitude but is reduced when the force points sideways/backward.
        v_cmd = self.apf_force_gain * force_norm
        v_cmd = float(np.clip(v_cmd, 0.0, self.v_max))
        v_cmd *= max(np.cos(force_offset), 0.0)

        if not avoidance_risk:
            # Limited avoidance mode: do not move far away from the goal unless collision is imminent.
            # During COLREG risk this limiter is relaxed so the required manoeuvre can develop.
            if abs(target_offset) > np.pi / 2:
                desired_turn_sign = np.sign(target_offset)

                if desired_turn_sign != 0.0 and np.sign(w_cmd) != desired_turn_sign:
                    w_cmd = 0.0
                    v_cmd = 0.0
                elif blocked_front:
                    v_cmd = 0.0
                elif abs(w_cmd) > 1e-6:
                    v_cmd = max(v_cmd, self.apf_turn_min_speed)
            else:
                target_component = abs(np.cos(target_offset))
                lateral_component = abs(np.sin(target_offset))

                if (
                    lateral_component > self.apf_deviation_gain * target_component
                    and lateral_component > 1e-6
                ):
                    v_cmd *= self.apf_deviation_gain * target_component / lateral_component

        # Apply acceleration limiting before converting body twist into thruster forces.
        max_delta_v = self.apf_a_max * max(self.lastdt, 1e-3)
        v_cmd = self.apf_prev_v_cmd + float(np.clip(v_cmd - self.apf_prev_v_cmd, -max_delta_v, max_delta_v))
        v_cmd = float(np.clip(v_cmd, 0.0, self.v_max))
        self.apf_prev_v_cmd = v_cmd
        if imminent_risk:
            self.apf_guidance_mode = "active_avoid"
        elif self.apf_colreg_active:
            self.apf_guidance_mode = "colreg_avoid"
        elif self.apf_side_lock_active:
            self.apf_guidance_mode = "side_lock_avoid"
        else:
            self.apf_guidance_mode = "goal"

        u_cmd[0, 0] = v_cmd
        u_cmd[1, 0] = w_cmd
        return u_cmd

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

            ### LIDAR + APF NAVIGATION CONTROL ##############
            t = self.timefromstart
            p_ref, u_ref, u_track = self.compute_route_tracking_control(t)
            final_distance = float(
                np.linalg.norm(
                    self.apf_final_goal_ne
                    - np.array([float(self.North), float(self.East)], dtype=float)
                )
            )

            if final_distance <= self.apf_goal_tolerance_m:
                # Stop only at the real final goal.
                self.u = Vector(2)
                self.apf_prev_v_cmd = 0.0
                self.apf_guidance_mode = "arrived"
                self.apf_force_body = np.zeros(2, dtype=float)
            elif self.apf_avoidance_needed():
                # APF is only active during obstacle avoidance.
                # Its attraction point follows the route ahead so it avoids while rejoining the route.
                p_apf, _ = self.s.p_u_sample(t + self.apf_route_lookahead_s)
                self.apf_goal_ne = np.array([float(p_apf[0, 0]), float(p_apf[1, 0])], dtype=float)
                u_apf = self.compute_apf_control()
                if self.apf_guidance_mode == "arrived":
                    # The look-ahead route point can be reached before the final goal; keep tracking.
                    self.u = u_track
                    self.apf_guidance_mode = "track"
                else:
                    self.u = Vector(2)
                    # During avoidance the route tracker can command negative surge
                    # while the boat is angled around the obstacle. Keep APF moving
                    # forward so the manoeuvre completes instead of spinning in place.
                    apf_surge = float(u_apf[0, 0])
                    if apf_surge > 1e-6:
                        self.u[0, 0] = max(apf_surge, self.apf_turn_min_speed)
                    else:
                        self.u[0, 0] = apf_surge
                    self.u[1, 0] = float(u_track[1, 0]) + self.apf_avoidance_turn_gain * float(u_apf[1, 0])
            else:
                # With no nearby obstacle, the robot follows the start-to-goal route directly.
                self.u = u_track
                self.apf_guidance_mode = "track"
                self.apf_force_body = np.zeros(2, dtype=float)

            self.u[1, 0] = np.clip(self.u[1, 0], -self.w_max, self.w_max)
            self.u[0, 0] = np.clip(self.u[0, 0], 0.0, self.v_max)
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
            print(
                'APF mode:', self.apf_guidance_mode,
                'encounter:', self.apf_encounter_mode,
                'bearing=', round(float(self.apf_encounter_bearing_deg), 1) if np.isfinite(self.apf_encounter_bearing_deg) else 'nan',
                'side=', int(self.apf_avoidance_side_sign),
                'DCPA=', round(float(self.apf_colreg_dcpa_m), 2) if np.isfinite(self.apf_colreg_dcpa_m) else 'nan',
                'COLREG=', self.apf_colreg_rule,
                'v=', round(float(v), 3),
                'w=', round(float(w), 3),
                'F_body=', np.round(self.apf_force_body, 3),
            )
            print('Prop rates: R=',self.right_rate,', L=',self.left_rate,'rad/s')

            if final_distance <= self.apf_goal_tolerance_m and np.isnan(self.s.t_complete):
                self.s.t_complete = self.timefromstart
            
        nearest_obstacle_north = np.nan
        nearest_obstacle_east = np.nan
        nearest_obstacle_distance = np.nan
        if self.nearest_lidar_obstacle is not None:
            centre_ne = np.asarray(
                self.nearest_lidar_obstacle.get("centre_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            nearest_obstacle_north = centre_ne[0]
            nearest_obstacle_east = centre_ne[1]
            nearest_obstacle_distance = float(
                self.nearest_lidar_obstacle.get(
                    "min_distance_m",
                    self.nearest_lidar_obstacle.get("distance_m", np.nan),
                )
            )

        ### LOG DATA ##############################
        with self.filename.open("a") as f:
            f.write(f"{current_epoch_s},{self.timefromstart},{self.right_rate},{self.left_rate},{self.lastdt},{self.Yaw},{self.North},{self.East},{self.sensed_yaw_rate},{self.integrated_yaw},{self.sensed_imu_stamp_s},{self.sensed_pos_northings_m},{self.sensed_pos_eastings_m},{self.sensed_pos_yaw_rad}, {self.sensed_pos_stamp_s}, {self.sensed_bottom_depth_stamp_s},{self.sensed_bottom_depth_m},{self.apf_guidance_mode},{self.apf_encounter_mode},{self.apf_avoidance_side_sign},{nearest_obstacle_north},{nearest_obstacle_east},{nearest_obstacle_distance},{self.apf_colreg_dcpa_m},{self.apf_colreg_tcpa_s}\n")
        
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
            
