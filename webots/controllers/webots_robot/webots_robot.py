import asyncio
import os
import ctypes
from math import atan2
import numpy as np
import json
import socket
from datetime import datetime, UTC
import traceback
import zmq
from zeroros import Subscriber, Publisher
from zeroros.messages import geometry_msgs, sensor_msgs, nav_msgs, range_bearing_msgs
from zeroros.message_broker import MessageBroker
from zeroros.rate import Rate
from controller import Supervisor
from controller.wb import wb
import copy

# Check if the platform is windows
if os.name == "nt":
    # Set the event loop policy to avoid the following warning:
    # [...]\site-packages\zmq\_future.py:681: RuntimeWarning:
    # Proactor event loop does not implement add_reader family of methods required for
    # zmq. Registering an additional selector thread for add_reader support via tornado.
    #  Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` to avoid
    # this warning.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def patch_message_broker():
    if getattr(MessageBroker, "_uos_safe_patch", False):
        return

    def safe_config_broker(self):
        frontend = None
        backend = None
        try:
            self.context = zmq.Context()
            frontend = self.context.socket(zmq.XSUB)
            frontend.bind("tcp://" + str(self.ip) + ":" + str(self.xsub_port))
            backend = self.context.socket(zmq.XPUB)
            backend.bind("tcp://" + str(self.ip) + ":" + str(self.xpub_port))
            zmq.proxy(frontend, backend)
        except zmq.error.ContextTerminated:
            print("Context terminated")
        except Exception as e:
            print("Error: ", e)
        finally:
            if frontend is not None:
                frontend.close()
            if backend is not None:
                backend.close()

    def safe_stop(self):
        context = getattr(self, "context", None)
        if context is not None:
            context.term()

    MessageBroker.config_broker = safe_config_broker
    MessageBroker.stop = safe_stop
    MessageBroker._uos_safe_patch = True


def is_port_open(ip, port, timeout=0.2):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        return sock.connect_ex((ip, port)) == 0
    finally:
        sock.close()


def broker_is_running(ip, xsub_port=5555, xpub_port=5556):
    return is_port_open(ip, xsub_port) and is_port_open(ip, xpub_port)

def Vector(n_rows): return np.zeros((n_rows,1), dtype=float)
def Matrix(n_rows, n_cols): return np.zeros((n_rows, n_cols), dtype=float)
def Identity(dim): return np.eye(dim, dtype=float)

# Transpose, Inverse matrix operations
def Transpose(m): return m.T
def Inverse(m): return np.linalg.inv(m)

# v2t adds a dimension of 1 on the bottom of a vector and t2v removes the last element of a vector
def v2t(x): return np.insert(x, len(x), 1, axis=0)
def t2v(x): return x[0:len(x)-1]

def l2m(nlist):
    # converts a list into a vector, or a list or lists into a matrix
    # this if checks if we have a list (it returns False) or a list of lists (returns True)
    if all(isinstance(item, list) for item in nlist) == False:  
        N = Vector(len(nlist[:]))
        for i in range(len(nlist[:])):
            N[i] = nlist[i]
    else: 
        N = Matrix(len(nlist[0]),len(nlist[:]))
        
        for i in range(len(nlist[:])):
            for j in range(len(nlist[i])):
                N[j,i] = nlist[i][j]                            
    return N

def TAM(phi,x,y): 
    # This function computes the thrust allocation matrix
    # Inputs:
    #   phi: [Nx1] vector of thruster axis angles w.r.t b-frame primary axis, degrees
    #   x: [Nx1] vector of thruster x offsets w.r.t b-frame origin, m
    #   y: [Nx1] vector of thruster y offsets w.r.t b-frame origin, m
    # Output:
    #   G: [3xN] matrix to convert thrust values to the b-frame x, y forces (N) and z moment (Nm)
    
    G=Matrix(3,len(phi))*np.nan    # initialise

    phi=np.deg2rad(phi) #convert degrees to radians
    
    for i in range(len(phi)): #can handle any number of thrusters
        G[0,i]=np.cos(phi[i][0])
        G[1,i]=np.sin(phi[i][0])
        # compute lever arms (note sin, cos to get orthogonal component)
        G[2,i]=x[i][0]*np.sin(phi[i][0])-y[i][0]*np.cos(phi[i][0])            

    return G


class HomogeneousTransformation:
 

    def __init__(self, t=Vector(2), gamma=0):
        # Note that the class implementation ONLY stores the translation vector and
        # rotation angle. The rotation matrix and homogeneous transformation matrix are
        # computed on the fly.
        self._t = t
        self._gamma = gamma

    def _check_homogeneous(self, H):
        if H.shape != (3, 3):
            raise ValueError("H must be a 3x3 matrix")
        if H[2, 2] != 1.0:
            raise ValueError("H must be a homogeneous matrix")

    def _check_rotation(self, R):
        if R.shape != (2, 2):
            raise ValueError("R must be a 2x2 matrix")
        if not np.allclose(np.linalg.det(R), 1.0, rtol=1e-2):
            raise ValueError(
                "R must be a rotation matrix. Determinant is not 1.0, but",
                np.linalg.det(R),
                "difference is",
                np.linalg.det(R) - 1.0,
            )

    @property  # automatically populates elements if the class is called
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property  # automatically populates elements if the class is called
    def R(self):
        R = Matrix(2, 2)

        gamma = self.gamma
        if isinstance(gamma, np.ndarray):
            gamma = gamma.item()   # extracts scalar from single-element array

        c = np.cos(gamma)
        s = np.sin(gamma)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        return R

    @R.setter
    def R(self, R):
        self._check_rotation(R)
        gamma_atan1 = np.arctan2(R[1, 0], R[0, 0])
        gamma_atan2 = np.arctan2(-R[0, 1], R[1, 1])
        if np.allclose(gamma_atan1, gamma_atan2):
            self.gamma = gamma_atan1
        else:
            raise ValueError(
                "R must be a rotation matrix. gamma is not the same for both arctan2"
            )

    @property  # automatically populates elements if the class is called
    def H(self):
        H = Identity(3)
        H[:2, :2] = self.R
        H[:2, 2:3] = self.t
        return H

    @H.setter
    def H(self, H):
        self._check_homogeneous(H)
        self.t = H[:2, 2:3]
        self.R = H[:2, :2]

    @property  # automatically populates elements if the class is called
    def H_R(self):
        H_R = Identity(3)
        H_R[:2, :2] = self.R
        return H_R

    @H_R.setter
    def H_R(self, H_R):
        self._check_homogeneous(H_R)
        self.R = H_R[:2, :2]

    @property  # automatically populates elements if the class is called
    def H_T(self):
        H_T = Identity(3)
        H_T[:2, 2:3] = self.t
        return H_T

    @H_T.setter
    def H_T(self, H_T):
        self._check_homogeneous(H_T)
        self.t = H_T[:2, 2:3]

    @property  # automatically populates elements if the class is called
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t              

class Vehicle2D_e:
    ###################################################
    # 
    # Wrapper class that defines platform properties 
    # and parses functions for translation and rotation
    # dynamics in the e-frame
    # Member attributes:
    #     m_tot: float, mass + added mass of the vehicle in surge, kg - we assume the vehicle moves in surge
    #     I_tot: float, inertia and added inertia for yaw, kgm 
    #     k_drag: float, quadratic non-linear drag doefficient for surge, kg/m - we assume the vehicle moves in surge
    #     B_66: float, # rotational drag in yaw
    #
    #  Member function:
    #     info(): # returns member attributes
    #     model(): # updates velocity vector
    #         Inputs:
    #              v: Previous velocity in the inertial frame as a [3x1] vector of floats
    #                  [[v_n]    velocity in north direction, m/s (fixed e-frame)
    #                  [v_e]    velocity in east direction, m/s (fixed e-frame)
    #                  [w]]     angular yaw rate, rad/s (fixed e-frame, optional)    
    #              trans: function for translation dynamics in e-frame, 
    #                 Inputs: T: Thrust in the inertial frame as a [2x1] or [3x1] vector of floats
    #                         v: Previous velocity in the inertial frame as a [2x1] or [3x1] vector of floats
    #                         dt: (optional) Time step, s, to progress v by, assumes 1 if not specified
    #                         k: (optional) drag coefficient, kg/m, of body assumes 1 if not specified
    #                         m: (optional) total mass, kg (mass+added), of body assumes 1 if not specified
    #                 Output: v_: updated velocity. Adopts shape of v
    #              rot: function for translation dynamics in e-frame, 
    #                 Inputs: T: Thrust in the inertial frame as a [2x1] or [3x1] vector of floats
    #                         v: Previous velocity in the inertial frame as a [2x1] or [3x1] vector of floats
    #                         dt: (optional) Time step, s, to progress v by, assumes 1 if not specified
    #                         B: (optional) rotational drag coefficient, kgm/s, of body assumes 1 if not specified
    #                         I: (optional) total moment of inertia, kgm (inertia+added), of body assumes 1 if not specified
    #                 Output: v_: updated velocity. Adopts shape of v
    #          Outputs: v_tr : updated velocity after translation and rotation
    ###################################################
    def __init__(self, m_tot, I_tot, k_drag, B_66):  # __init__ is the class constructor
        self.m_tot = m_tot      
        self.I_tot = I_tot      
        self.k_drag = k_drag    
        self.B_66 = B_66  
    
    def info(self):          # info is a member function
        print('Platform parameters:',
              '\nm_tot =', self.m_tot,'kg',
              '\nk_drag=',self.k_drag,'kg/m',
              '\nI_tot=',self.I_tot,'kgm^2',
              '\nB_66 =', self.B_66,'kgm/s')

    def model(self,trans,rot,T,v,dt=1): # models dynamics using translate and rotational functions
        
        v_t = trans(T, v, dt=dt, k=self.k_drag, m=self.m_tot)
        v_tr = rot(T, v_t, dt=dt, B=self.B_66, I=self.I_tot)

        return v_tr

def dynamics_translation_e(T, v, dt=1, k=1, m=1):
    ################################################# 
    # This function models drag to determines rigid body velocity in the e-frame
    # It solves equations of the form
    #  a = (T + F_d)/m    
    # and progresses velocity v to the next timestep v_ 
    # based on acceleration 'a' acting for the period 'dt'. 
    #  v = v+a.dt   
    #
    # Inputs
    #   T: Thrust in the inertial frame as a [2x1] or [3x1] vector of floats
    #          [[T_n]    Thrust in north direction, N (fixed e-frame)
    #           [T_e]    Thrust in east direction, N (fixed e-frame)
    #           [tau_z]]  Moment in yaw, Nm (fixed e-frame, optional - ignored)
    #   v: Previous velocity in the inertial frame as a [2x1] or [3x1] vector of floats
    #          [[v_n]    velocity in north direction, m/s (fixed e-frame)
    #           [v_e]    velocity in east direction, m/s (fixed e-frame)
    #           [w]]     angular yaw rate, rad/s (fixed e-frame, optional - ignored)    
    #   dt: (optional) Time step, s, to progress v by, assumes 1 if not specified
    #   k: (optional) drag coefficient, kg/m, of body assumes 1 if not specified
    #   m: (optional) total mass, kg (mass+added), of body assumes 1 if not specified
    #
    # Output
    #  v_: updated velocity. Adopts shape of v
    #
    # Dependancies
    #    standard python library numpy and copy
    ###################################################    
            
    # Compute drag force
    speed = np.sqrt(v[0][0]**2 + v[1][0]**2)
    fdrag = k*speed**2
    
    # Compute N- and E-components of drag using similar triangles 
    # (computationally more efficient that trigonometry)
    if speed > 0:
        fdrag_n = -(v[0][0] / speed) * fdrag # identical to -fdrag * cos(gamma), where gamma = atan2(v_east, v_north)
        fdrag_e  = -(v[1][0] / speed)  * fdrag # identical to -fdrag * sin(gamma), where gamma = atan2(v_east, v_north)       
    else:
        fdrag_n = 0        
        fdrag_e  = 0
        
    # Isolate acceleration
    a_n = (T[0][0] + fdrag_n) / m
    a_e  = (T[1][0]  + fdrag_e) / m
    
    # New velocity
    v_=copy.copy(v) #copy shape and values of previous velocity, use this to avoid updating v
    v_[0][0] += a_n * dt
    v_[1][0] += a_e * dt
    
    return v_


def dynamics_rotation_e(T, v, dt=1, B=1, I=1):
    ################################################# 
    # This function models drag to determines rigid body velocity in the e-frame
    # It solves equations of the form
    #  a = (tau_z + tau_drag)/I    
    # and progresses angular velocity component of v to the next timestep v_ 
    # based on acceleration 'a' acting for the period 'dt'. 
    #  v = v+a.dt   
    #
    # Inputs
    #   T: Thrust in the inertial frame as a scalar or [3x1] vector of floats
    #          [[T_n]    Thrust in north direction, N (fixed e-frame)
    #           [T_e]    Thrust in east direction, N (fixed e-frame)
    #           [tau_z]]  Moment in yaw, Nm (fixed e-frame, optional - ignored)
    #
    #   v: Previous velocity in the inertial frame as a [3x1] vector of floats
    #          [[v_n]    velocity in north direction, m/s (fixed e-frame)
    #           [v_e]    velocity in east direction, m/s (fixed e-frame)
    #           [w]]     angular yaw rate, rad/s (fixed e-frame, optional - ignored)    
    #   dt: (optional) Time step, s, to progress v by, assumes 1 if not specified
    #   B: (optional) rotational drag coefficient, kgm/s, of body assumes 1 if not specified
    #   I: (optional) total moment of inertia, kgm (inertia+added), of body assumes 1 if not specified
    #
    # Output
    #  v_: updated velocity. Only updates w
    #
    # Dependancies
    #    standard python library numpy and copy
    ###################################################    
            
    # Compute drag force    
    
    # Compute N- and E-components of drag using similar triangles 
    # (computationally more efficient that trigonometry)
    tau_drag = -(v[2][0]*B)        
        
    # Isolate acceleration
    a = (T[2][0] + tau_drag) / I
        
    # New rate
    v_=copy.copy(v) #copy shape and values of previous velocity, use this to avoid updating v
    v_[2][0] += a * dt
    
    return v_


def rigid_body_kinematics(mu,u,dt=0.1):
    #################################################################################
    # This function models the forwards kinematics of rigid body, b
    # It implements a model of form
    #  mu_k = f(mu_k-1,u_k)
    # and progresses the pose of the robot 'mu' (in the fixed frame e)) 
    # to the next timestep 'mu_' by applying combined linear and angular velocity (twist) 
    # as its control action 'u' for the period 'dt'. 
    #
    # The notation of 'mu' is used as it represents the mean estimate of the robot pose p
    #
    # It implements the homogeneous transformation below to achieve this.
    # 
    # Xk = Xk-1 Trans(tbc)Rot(wdt)Trans(tcb')
    #
    # as 
    #
    # H_eb' = H_eb @ H_bc.H_T @ H_cb'.H_R @ H_cb'.H_T
    #
    # Inputs
    #.  mu: The previous pose p_{k-1}, which is a [3x1] matrix of form
    #          [[x]     Northings in m (fixed frame, e )
    #           [y]     Eastings in m (fixed frame, e )
    #           [g]]    Heading in rads (fixed frame, e )
    #.  u:  The control is a [2 x 1] matrix of the form, 
    #         [[v]      Linear forwards velocity in m/s (body frame, b)
    #.         [w]]     Angular velocity in rads/s (body frame, b)
    # Output
    #.  mu: The new pose p_k
    #
    # Dependancies 
    #.  standard python libraries (numpy)
    #.  class HomogeneousTransformation from course library `math_feeg6043.py`
    #################################################################################    
    tol = 1e-4 # to avoid numerical instabilities, anything smaller than this we treat as zero
    
    # create an empty containor for the output
    mu_=Vector(3)

    # create an empty containor for the output homogeneous transformation matrix     
    H_eb_ = HomogeneousTransformation()
    
    # convert input pose p_k-1 into a homogeneous transformation matrix 
    t_eb=Vector(2)
    t_eb[0] = mu[0] # northings (p_k-1)
    t_eb[1] = mu[1] #eastings (p_k-1)
    g_eb = mu[2] #heading (p_k-1)
    H_eb = HomogeneousTransformation(t_eb,g_eb)            

    if abs(u[0])<tol and abs(u[1])<tol:
        # handles the stationary case where 
        H_eb_ = H_eb

    elif abs(u[1])<tol:        
        # implement a simpler vesion of the model that doesn't need to compute twist 
        # H_eb_ = H_eb@H_bb' 

        v = u[0] #surge rate        
            
        # compute motion in the body frame due to the pure linear velocity
        t_bb_=Vector(2) # [2x1] matrix of 0
        t_bb_[0] = v*dt
        
        # create the homogeneous transformation from b to b', 
        H_bb_ = HomogeneousTransformation(t_bb_,0) # heading doesn't change as w=0 (u[1]=0]
        
        # left multiply be the homogeneous transformation from the fixed frame e to the body frame b
        H_eb_.H = H_eb.H@H_bb_.H            
            
    else: 
        # implements the model derived for twist
        
        v = u[0] #surge rate
        w = u[1] #yaw rate

        # calculate centre of rotation from the initial body position
        t_bc = Vector(2) # [2x1] matrix of 0                       
        t_bc[1]=v/w      # centre of rotation is v/w in the +ve y direction of the body Eq(A1.2.12)
        
        # the centre of rotation 'c' keeps the heading of the body, so is 0 as seen from the body
        H_bc = HomogeneousTransformation(t_bc,0)
            
        # to rotate the body about 'c' so need 'b' as seen from the centre of rotation
        H_cb = HomogeneousTransformation() 
        H_cb.H = Inverse(H_bc.H)  

        # to rotate the body b around the centre of rotation w*dt while maintain the same radius
        H_cb_ = HomogeneousTransformation(H_cb.t,w*dt)
        
        # we rotate first, and then translate
        H_eb_.H = H_eb.H@H_bc.H@H_cb_.H_R@H_cb_.H_T                      
        
    # create the pose, handling angle wrap at 2pi (in rads)
    mu_[0] = H_eb_.t[0]
    mu_[1] = H_eb_.t[1]
    mu_[2] = H_eb_.gamma % (2 * np.pi ) #(H_eb_.gamma + np.pi) % (2 * np.pi ) - np.pi 
    
    return mu_


class UDPBroadcastServer:
    def __init__(self, ip, port):
        # -- Enable port reusage
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        # -- Enable broadcasting mode if feature is available
        if hasattr(socket, "SO_REUSEPORT"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # -- Enable broadcasting mode
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.socket.settimeout(None)
        self.ip = ip
        self.port = port
        

    def broadcast(self, message):
        # -- Broadcast the dictionary as a bytes-like object (string-like info) and empties
        broadcast_string = json.dumps(message, indent=3)
        self.socket.sendto(
            broadcast_string.encode("utf-8"),
            (self.ip, self.port),
        )

    def close(self):
        try:
            self.socket.close()
        except Exception:
            pass


class WebotsController(Supervisor):
    def __init__(self):
        
        super(WebotsController, self).__init__()
        self._shutdown_done = False
        self.ip = "127.0.0.1"
        self.port = 5600
        self.prop_distance = 0.19        
        self.marker_id = 0

        self.start_time = datetime.now(UTC).timestamp()
        patch_message_broker()
        self.broker = None
        if broker_is_running(self.ip):
            print("ZeroROS broker already running; reusing existing broker.")
        else:
            self.broker = MessageBroker(ip=self.ip)

        self.groundtruth_pub = Publisher(
            "/groundtruth", geometry_msgs.PoseStamped
        )        
        self.control_sub = Subscriber("/control", geometry_msgs.Vector3, self.control_callback)
        self.gyro_pub = Publisher(
            "/imu", geometry_msgs.Vector3
        )

        self.sonar_pub = Publisher("/sonar", geometry_msgs.Vector3)
        self.laserscan_pub = Publisher("/lidar", range_bearing_msgs.RBLaserScan)


        # UDP server to fake Aruco marker detection
        self.udp_server = UDPBroadcastServer("127.0.0.1", 50000)
        self.udp_rate = Rate(1)

        timestep = int(self.getBasicTimeStep())
        self.timeStep = timestep*1
        print("Timestep:", timestep, "Setting controller timestep: ", self.timeStep)


        self.pose_msg = geometry_msgs.PoseStamped()

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timeStep)

        self.imu = self.getDevice("gyro")
        self.imu.enable(self.timeStep)
        self.prev_yaw = None
        self.prev_time = 0

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timeStep)

        self.num_lidar_msgs = 0
        self.lidar = None
        self.last_lidar_timestamp = None
        self.lidar_rate = None
        self.lidar_msg = None
        try:
            self.lidar = self.getDevice("lidar")
            if self.lidar is None:
                raise RuntimeError("LiDAR device 'lidar' not found")

            self.lidar.enable(self.timeStep)
            self.lidar.enablePointCloud()
            self.lidar_rate = Rate(self.lidar.getFrequency())
            self.lidar_msg = range_bearing_msgs.RBLaserScan()
            self.lidar_msg.header.frame_id = "lidar"
            self.lidar_msg.angle_min = -self.lidar.getFov() / 2.0
            self.lidar_msg.angle_max = self.lidar.getFov() / 2.0
            self.lidar_msg.angle_increment = (
                self.lidar.getFov() / self.lidar.getHorizontalResolution()
            )
            self.lidar_msg.time_increment = self.lidar.getSamplingPeriod() / (
                1000.0 * self.lidar.getHorizontalResolution()
            )
            self.lidar_msg.scan_time = self.lidar.getSamplingPeriod() / 1000.0
            self.lidar_msg.range_min = self.lidar.getMinRange()
            self.lidar_msg.range_max = self.lidar.getMaxRange()
            self.lidar_msg.intensities = np.array([0] * len(self.lidar_msg.ranges))
            self.lidar_msg.angles = np.array(
                [
                    self.lidar_msg.angle_min + i * self.lidar_msg.angle_increment
                    for i in range(len(self.lidar_msg.ranges))
                ]
            )
            print(
                "LiDAR initialised:"
                f" fov={self.lidar.getFov():.4f} rad,"
                f" res={self.lidar.getHorizontalResolution()},"
                f" freq={self.lidar.getFrequency():.2f} Hz"
            )
        except Exception as e:
            self.lidar = None
            self.lidar_rate = None
            self.lidar_msg = None
            print("Warning: LiDAR init failed, running without /lidar publishing:", e)

        # Propulsion geometry expressed in the robot body frame.
        self.right_thruster_offset = np.array(
            [-0.1, -self.prop_distance / 2.0, -0.05],
            dtype=float,
        )
        self.left_thruster_offset = np.array(
            [-0.1, self.prop_distance / 2.0, -0.05],
            dtype=float,
        )
        self._vector3_type = ctypes.c_double * 3

        # State mirrored from the Webots physics body.
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
        }
        self.twist = {
            "surge": 0.0,
            "sway": 0.0,
            "heave": 0.0,
            "yaw_rate": 0.0,
        }
        self.control = {
            "right_rate": 0.0,
            "left_rate": 0.0,
        }

        self.supervisor_node = None
        self.translation_field = None
        self.rotation_field = None
        if self.supervisor:
            self.supervisor_node = self.getSelf()
            self.translation_field = self.supervisor_node.getField("translation")
            self.rotation_field = self.supervisor_node.getField("rotation")
            self._sync_ego_state_from_world()
            print(
                "Initial ego pose from world:",
                self.state["x"],
                self.state["y"],
                self.state["yaw"],
            )

        self.moving_robot_started = False
        self.motion_targets = []
        self.motion_target_configs = {
            "MOVING_ROBOT": {
                "speed": 0.118,
                "acceleration": 0.080,
                "yaw_rate": 0.0,
                "turn_radius": 0.0,
                "lock_x": 3.5,
                "wrap_y_top": 2.6,
                "wrap_y_bottom": -3.8,
                "start_with_zero_speed": True,
            },
            "MOVING_ROBOT_2": {
                "speed": 0.118,
                "acceleration": 0.060,
                "yaw_rate": 0.0,
                "turn_radius": 0.0,
                "lock_x": 4.67,
                "wrap_y_top": 2.6,
                "wrap_y_bottom": -3.8,
                "start_with_zero_speed": True,
            },
            "FRONT_OBSTACLE_ROBOT": {
                "speed": 0.074,
                "acceleration": 0.025,
                "yaw_rate": 0.0,
                "turn_radius": 0.0,
                "lock_y": -1.0,
                "start_with_zero_speed": True,
            },
            "HEAD_ON_OBSTACLE_ROBOT": {
                "speed": 0.062,
                "acceleration": 0.035,
                "yaw_rate": 0.0,
                "turn_radius": 0.0,
                "lock_y": -1.0,
                "start_with_zero_speed": True,
            },
        }

        for target_name in ("MOVING_ROBOT", "MOVING_ROBOT_2"):
            node = self.getFromDef(target_name)
            if node is None:
                continue
            target = self._make_motion_target(
                target_name,
                node,
                **self.motion_target_configs[target_name],
            )
            if target is not None:
                self.motion_targets.append(target)

        front_obstacle_node = self.getFromDef("FRONT_OBSTACLE_ROBOT")
        head_on_obstacle_node = self.getFromDef("HEAD_ON_OBSTACLE_ROBOT")
        static_path_obstacle_node = self.getFromDef("STATIC_PATH_OBSTACLE_ROBOT")
        if front_obstacle_node is not None:
            target = self._make_motion_target(
                "FRONT_OBSTACLE_ROBOT",
                front_obstacle_node,
                **self.motion_target_configs["FRONT_OBSTACLE_ROBOT"],
            )
            if target is not None:
                self.motion_targets.append(target)
        elif head_on_obstacle_node is not None:
            target = self._make_motion_target(
                "HEAD_ON_OBSTACLE_ROBOT",
                head_on_obstacle_node,
                **self.motion_target_configs["HEAD_ON_OBSTACLE_ROBOT"],
            )
            if target is not None:
                self.motion_targets.append(target)
        elif static_path_obstacle_node is not None:
            print("Static path obstacle robot found; leaving it stationary.")
        else:
            print("Warning: no front obstacle robot found; skipping obstacle update.")

    def control_callback(self, control_message):   
        self.control["right_rate"] = control_message.x
        self.control["left_rate"]  = control_message.y
        if (
            not self.moving_robot_started
            and (abs(float(control_message.x)) > 1e-3 or abs(float(control_message.y)) > 1e-3)
        ):
            self.moving_robot_started = True

    def _shutdown_zeroros_endpoint(self, endpoint):
        if endpoint is None:
            return
        stop_fn = getattr(endpoint, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception as e:
                print("Warning: endpoint stop failed:", e)
        sock = getattr(endpoint, "sock", None)
        if sock is not None:
            try:
                sock.close(0)
            except Exception:
                pass
        context = getattr(endpoint, "context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        self._shutdown_zeroros_endpoint(getattr(self, "control_sub", None))
        self._shutdown_zeroros_endpoint(getattr(self, "groundtruth_pub", None))
        self._shutdown_zeroros_endpoint(getattr(self, "gyro_pub", None))
        self._shutdown_zeroros_endpoint(getattr(self, "sonar_pub", None))
        self._shutdown_zeroros_endpoint(getattr(self, "laserscan_pub", None))

        broker = getattr(self, "broker", None)
        if broker is not None:
            try:
                if getattr(broker, "context", None) is not None:
                    broker.stop()
            except Exception as e:
                print("Warning: broker stop failed:", e)
            broker_thread = getattr(broker, "broker_thread", None)
            if broker_thread is not None and broker_thread.is_alive():
                broker_thread.join(timeout=0.5)

        udp_server = getattr(self, "udp_server", None)
        if udp_server is not None:
            udp_server.close()

    @staticmethod
    def _wrap_to_pi(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _node_yaw(self, node):
        orientation = node.getOrientation()
        return float(atan2(orientation[3], orientation[0]))

    def _sync_ego_state_from_world(self):
        if self.supervisor_node is None:
            return

        position = np.array(self.supervisor_node.getPosition(), dtype=float)
        velocity = np.array(self.supervisor_node.getVelocity(), dtype=float)
        yaw = self._node_yaw(self.supervisor_node)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        linear_world = velocity[:3]

        self.state["x"] = float(position[0])
        self.state["y"] = float(position[1])
        self.state["z"] = float(position[2])
        self.state["yaw"] = float(yaw)
        self.twist["surge"] = float(cos_yaw * linear_world[0] + sin_yaw * linear_world[1])
        self.twist["sway"] = float(-sin_yaw * linear_world[0] + cos_yaw * linear_world[1])
        self.twist["heave"] = float(linear_world[2])
        self.twist["yaw_rate"] = float(velocity[5])

    def _add_relative_force_with_offset(self, force, offset):
        if self.supervisor_node is None:
            return

        wb.wb_supervisor_node_add_force_with_offset(
            self.supervisor_node._ref,
            self._vector3_type(*[float(v) for v in force]),
            self._vector3_type(*[float(v) for v in offset]),
            1,
        )

    def _make_motion_target(
        self,
        name,
        node,
        speed,
        acceleration=0.0,
        yaw_rate=0.0,
        turn_radius=0.0,
        lock_x=None,
        lock_y=None,
        wrap_y_top=None,
        wrap_y_bottom=None,
        start_with_zero_speed=False,
    ):
        if node is None:
            return None

        translation_field = node.getField("translation")
        rotation_field = node.getField("rotation")
        position = np.array(translation_field.getSFVec3f(), dtype=float)
        initial_position = position.copy()
        initial_yaw = self._node_yaw(node)

        current_speed = 0.0 if start_with_zero_speed and acceleration > 0.0 else float(speed)
        return {
            "name": name,
            "translation_field": translation_field,
            "rotation_field": rotation_field,
            "position": position,
            "initial_position": initial_position,
            "yaw": initial_yaw,
            "initial_yaw": initial_yaw,
            "speed": float(speed),
            "current_speed": float(current_speed),
            "acceleration": float(acceleration),
            "yaw_rate": float(yaw_rate),
            "turn_radius": float(turn_radius),
            "lock_x": lock_x,
            "lock_y": lock_y,
            "wrap_y_top": wrap_y_top,
            "wrap_y_bottom": wrap_y_bottom,
        }

    @staticmethod
    def _advance_speed(current_speed, target_speed, acceleration, dt):
        current_speed = float(current_speed)
        target_speed = float(target_speed)
        acceleration = float(acceleration)
        dt = float(dt)

        if acceleration <= 1e-9:
            return target_speed

        delta_speed = target_speed - current_speed
        max_step = acceleration * dt
        if abs(delta_speed) <= max_step:
            return target_speed
        return current_speed + np.sign(delta_speed) * max_step

    def _apply_motion_target(self, target, dt):
        target["current_speed"] = self._advance_speed(
            target["current_speed"],
            target["speed"],
            target["acceleration"],
            dt,
        )

        yaw_rate = float(target["yaw_rate"])
        turn_radius = float(target["turn_radius"])
        if abs(yaw_rate) <= 1e-9 and abs(turn_radius) > 1e-9 and abs(target["current_speed"]) > 1e-9:
            yaw_rate = target["current_speed"] / turn_radius

        target["yaw"] = self._wrap_to_pi(target["yaw"] + yaw_rate * dt)
        direction = np.array(
            [np.cos(target["yaw"]), np.sin(target["yaw"]), 0.0],
            dtype=float,
        )
        target["position"] = target["position"] + direction * target["current_speed"] * dt

        if target["lock_x"] is not None and abs(yaw_rate) <= 1e-9 and abs(turn_radius) <= 1e-9:
            target["position"][0] = float(target["lock_x"])
        if target["lock_y"] is not None and abs(yaw_rate) <= 1e-9 and abs(turn_radius) <= 1e-9:
            target["position"][1] = float(target["lock_y"])

        wrap_y_top = target["wrap_y_top"]
        wrap_y_bottom = target["wrap_y_bottom"]
        if wrap_y_top is not None and wrap_y_bottom is not None:
            if target["position"][1] < wrap_y_bottom:
                target["position"] = target["initial_position"].copy()
                target["position"][1] = float(wrap_y_top)
                target["yaw"] = float(target["initial_yaw"])
                target["current_speed"] = 0.0 if target["acceleration"] > 0.0 else float(target["speed"])
            elif target["position"][1] > wrap_y_top:
                target["position"] = target["initial_position"].copy()
                target["position"][1] = float(wrap_y_bottom)
                target["yaw"] = float(target["initial_yaw"])
                target["current_speed"] = 0.0 if target["acceleration"] > 0.0 else float(target["speed"])

        target["translation_field"].setSFVec3f(target["position"].tolist())
        if target["rotation_field"] is not None:
            target["rotation_field"].setSFRotation([0, 0, 1, float(target["yaw"])])
          
    def run(self): 
        try:
            while self.step(self.timeStep) != -1:
                try:
                    self.infinite_loop()
                except Exception as e:
                    print("Warning: controller loop error:", e)
                    traceback.print_exc()
        finally:
            self.shutdown()

    def rpm2N(self, x, fwd_lim = 2000, rev_lim = -2000): 
        tol = 10                
        if x>fwd_lim: x=fwd_lim
        if x<rev_lim: x=rev_lim
        if abs(x) <= tol: return 0
        elif x>tol: return 1.542E-7*x**2+0.0003293*x-0.001401
        else: return -7.357E-8*x**2+0.0001717*x-1.054E-16
        

    def dynamics_engine(self):
        if self.supervisor_node is None:
            return

        self._sync_ego_state_from_world()

        # Apply thrust directly to the rigid body so Fluid handles buoyancy and drag.
        right_force = self.rpm2N(-self.control["right_rate"] * 60 / (2 * np.pi))
        left_force = self.rpm2N(self.control["left_rate"] * 60 / (2 * np.pi))
        self._add_relative_force_with_offset([right_force, 0.0, 0.0], self.right_thruster_offset)
        self._add_relative_force_with_offset([left_force, 0.0, 0.0], self.left_thruster_offset)

    def infinite_loop(self):    
        self.update_moving_robot()

        self.dynamics_engine()

        current_time = datetime.now(UTC).timestamp()
        if (
            self.lidar is not None
            and self.lidar_rate is not None
            and self.lidar_msg is not None
            and self.lidar_rate.remaining() <= 0
        ):
            self.last_lidar_timestamp = current_time
            msg = self.lidar_msg
            msg.ranges = np.array(self.lidar.getRangeImage(), dtype=float)
            msg.ranges[msg.ranges == float("inf")] = 0.0
            msg.angles = np.array(
                [
                    msg.angle_min + i * msg.angle_increment
                    for i in range(len(msg.ranges))
                ],
                dtype=float,
            )
            msg.header.stamp = current_time
            msg.header.seq = self.num_lidar_msgs
            self.laserscan_pub.publish(msg)
            self.num_lidar_msgs += 1
            self.lidar_rate.reset()

        pose_val = self.gps.getValues()  
        pose_time = current_time
        elapsed_time = pose_time - self.start_time
        # Switch x and y
        pose_val = [-pose_val[1], pose_val[0], pose_val[2]]

        north = self.compass.getValues()

        # The Compass node returns a vector that indicates the north direction specified
        # by the coordinateSystem field of the WorldInfo node.
        # Transform from ENU to NED
        north = [north[1], north[0], -north[2]]

        ##change angle to clockwise
        angle = atan2(north[1], north[0])
        angle = ((np.pi*2) - angle) % (np.pi*2)
        if angle == (np.pi*2):
            angle = 0

        quaternion = geometry_msgs.Quaternion()
        quaternion.from_euler(0, 0, angle)

        # Publish groundtruth
        self.pose_msg.header.stamp = pose_time
        self.pose_msg.pose.position.x = pose_val[1]
        self.pose_msg.pose.position.y = pose_val[0]
        self.pose_msg.pose.position.z = pose_val[2]
        self.pose_msg.pose.orientation.x = quaternion.x
        self.pose_msg.pose.orientation.y = quaternion.y
        self.pose_msg.pose.orientation.z = quaternion.z
        self.pose_msg.pose.orientation.w = quaternion.w
        self.groundtruth_pub.publish(self.pose_msg)

        # Publish gryo
        # wx, wy, wz = self.imu.getValues()
        # print(f"wx={wx:.6f}, wy={wy:.6f}, wz={wz:.6f}")

        # Compute yaw rate as numerical derivative
        dt = max(current_time - self.prev_time, 1e-6)
        if self.prev_yaw is None: yaw_rate = 0
        else: 
            delta_yaw = angle - self.prev_yaw
            delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi

            yaw_rate = (delta_yaw) / dt

        self.prev_yaw = angle
        self.prev_time = current_time

        # Handle angle wrapping around 2*pi
        if yaw_rate > np.pi/dt:
            yaw_rate -= 2*np.pi/dt
        elif yaw_rate < -np.pi/dt:
            yaw_rate += 2*np.pi/dt

        self.gyro_msg = geometry_msgs.Vector3()        
        #self.gyro_msg.header.stamp = datetime.now(UTC).timestamp() - self.start_time
        self.gyro_msg.x = None
        self.gyro_msg.y = None
        self.gyro_msg.z = yaw_rate
        # print(self.timeStep)

        

        self.gyro_pub.publish(self.gyro_msg)

        echorange = geometry_msgs.Vector3()  
        echorange.x = None
        echorange.y = None
        echorange.z = 1000+np.random.uniform(-100,100)

        self.sonar_pub.publish(echorange)

        if self.udp_rate.remaining() <= 0:
            # Fake Aruco marker detection
            # Broadcast the pose of the robot
            msg = {}
            msg[self.marker_id] = [
                pose_time,
                elapsed_time,
                pose_val[1],
                pose_val[0],
                pose_val[2],
                0.0,
                0.0,
                angle,
            ]
            self.udp_server.broadcast(msg)
            print('Elapsed time [s]:', elapsed_time)
            print('Aruco: N,E,G [m,m,rad]:', pose_val[1], pose_val[0], angle)
            print('Gyro: yaw [rad/s]:', (self.gyro_msg.z))
            print('Control: right_prop, left_prop [rad/s]:', self.control["right_rate"], self.control["left_rate"])
            self.udp_rate.reset()

    def update_moving_robot(self):
        if not self.moving_robot_started:
            return

        dt = self.timeStep / 1000.0
        for target in self.motion_targets:
            self._apply_motion_target(target, dt)


wc = WebotsController()
try:
    wc.run()
finally:
    wc.shutdown()
