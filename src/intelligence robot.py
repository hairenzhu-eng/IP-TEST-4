import numpy as np
import argparse
import csv
from datetime import datetime
from pathlib import Path

from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import RBLaserScan, Vector3Stamped, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

from math_feeg6043 import Vector
from model_feeg6043 import (
    RangeAngleKinematics,
    ParticlePathSLAM,
    ActuatorConfiguration,
    rigid_body_kinematics,
)


ARUCO_NOISE_ENABLED = True
ARUCO_XY_STD = 0.02          # metres
ARUCO_YAW_STD_DEG = 2.0      # degrees
ARUCO_NOISE_SEED = 11


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def angle_error(target, current):
    return wrap_angle(target - current)


# ----------------------------------------------------------------------
# Assignment experiment presets
# ----------------------------------------------------------------------
EXPERIMENT_PRESETS = {
    "baseline": {
        "n_particles": 10,
        "process_v_std": 0.01,
        "process_w_std_deg": 1.5,
        "lidar_std": 0.08,
        "map_std": 0.08,
        "description": "Medium process noise and medium map observation noise.",
    },
    "low_process": {
        "n_particles": 10,
        "process_v_std": 0.005,
        "process_w_std_deg": 0.5,
        "lidar_std": 0.08,
        "map_std": 0.08,
        "description": "Low process noise with fixed map observation noise.",
    },
    "high_process": {
        "n_particles": 10,
        "process_v_std": 0.03,
        "process_w_std_deg": 5.0,
        "lidar_std": 0.08,
        "map_std": 0.08,
        "description": "High process noise with fixed map observation noise.",
    },
    "low_map_noise": {
        "n_particles": 10,
        "process_v_std": 0.01,
        "process_w_std_deg": 1.5,
        "lidar_std": 0.03,
        "map_std": 0.03,
        "description": "Low LiDAR/map observation noise.",
    },
    "high_map_noise": {
        "n_particles": 10,
        "process_v_std": 0.01,
        "process_w_std_deg": 1.5,
        "lidar_std": 0.15,
        "map_std": 0.15,
        "description": "High LiDAR/map observation noise.",
    },
    "low_particles": {
        "n_particles": 5,
        "process_v_std": 0.01,
        "process_w_std_deg": 1.5,
        "lidar_std": 0.08,
        "map_std": 0.08,
        "description": "Low particle count sensitivity test.",
    },
    "high_particles": {
        "n_particles": 15,
        "process_v_std": 0.01,
        "process_w_std_deg": 1.5,
        "lidar_std": 0.08,
        "map_std": 0.08,
        "description": "High particle count sensitivity test.",
    },
}


class LaptopPilot:
    def __init__(self, simulation=True, experiment="high_particles", laps=1):
        self.simulation = simulation
        self.experiment = experiment
        self.required_laps = int(laps)

        if experiment not in EXPERIMENT_PRESETS:
            raise ValueError(f"Unknown experiment preset: {experiment}")

        self.preset = EXPERIMENT_PRESETS[experiment]

        if self.simulation:
            aruco_params = {"port": 50000, "marker_id": 0}
            self.robot_ip = "127.0.0.1"
        else:
            aruco_params = {"port": 50002, "marker_id": 21}
            self.robot_ip = "192.168.90.1"

        print("Connecting to robot with IP", self.robot_ip)
        print("Simulation mode:", self.simulation)
        print("Experiment:", self.experiment)
        print("Experiment description:", self.preset["description"])

        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)
        self.aruco_rng = np.random.default_rng(ARUCO_NOISE_SEED)

        # ---------------- Visualiser variables ----------------
        self.northings_path = []
        self.eastings_path = []
        self.corners = []

        # SLAM rate limiter. This prevents ParticlePathSLAM/GPR from becoming too heavy.
        self.last_slam_update_time = 0.0
        self.slam_update_period_s = 0.5

        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        self.turn_direction = -1.0
        self.turn_start_time = None
        self.max_turn_time_s = 3.0

        self.last_corner_time = 0.0
        self.lap_corner = 20

        self.est_pose_northings_m = 0.0
        self.est_pose_eastings_m = 0.0
        self.est_pose_yaw_rad = 0.0

        self.lidar_data = None
        self.lidar_data_rb = None
        self.lidar_timestamp_s = None
        self.lidar_new = False

        # ---------------- State ----------------
        self.initialise_pose = False
        self.aruco_used_for_initial_pose = False

        self.t_prev = datetime.utcnow().timestamp()
        self.stop_flag = False
        self.sim_time_offset = 0
        self.sim_init = simulation

        # ---------------- Robot model ----------------
        wheel_distance = 0.081
        wheel_diameter = 0.074
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter)

        self.cmd_wheelrate_right = 0.0
        self.cmd_wheelrate_left = 0.0
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None

        # Physical robot: measured wheel speeds are more defensible.
        # Simulation: command input is often smoother if wheel feedback is delayed.
        self.use_command_for_odometry = bool(self.simulation)
        self.last_u_cmd = Vector(2)
        self.last_u_cmd[0] = 0.0
        self.last_u_cmd[1] = 0.0

        # ---------------- LiDAR + Particle Path SLAM ----------------
        self.lidar = RangeAngleKinematics(0.1, 0.0)

        self.n_particles = int(self.preset["n_particles"])
        self.particle_slam = ParticlePathSLAM(
            self.n_particles,
            self.lidar,
            position_std=0.02,
        )

        self.rbf_length_scale = np.deg2rad(20)
        self.lidar_std = float(self.preset["lidar_std"])
        self.map_std = float(self.preset["map_std"])
        self.gpr_sample_cap = 5
        self.max_lidar_points_for_slam = 25

        self.process_v_std = float(self.preset["process_v_std"])
        self.process_w_std = np.deg2rad(float(self.preset["process_w_std_deg"]))

        self.position_uncertainty_m = 0.0
        self.particle_northings = []
        self.particle_eastings = []

        self.initialise_particles()

        # ---------------- Behaviour controller ----------------
        self.v_forward = 0.08

        # Nearly-in-place turning gives cleaner 90-degree behaviour.
        self.v_turn = 0.07
        self.w_turn = 0.30

        # Do not turn because of far-away features.
        self.front_angle_limit = np.deg2rad(25)
        self.front_block_threshold = 0.5
        self.min_front_close_beams = 5

        self.v_max = 0.4
        self.w_max = 0.8

        self.drive_state = "STRAIGHT"
        self.target_yaw_rad = None
        self.turn_tolerance = np.deg2rad(6)

        self.side_angle_min = np.deg2rad(45)
        self.side_angle_max = np.deg2rad(120)

        self.turn_cooldown_until = 0.0
        self.turn_cooldown_s = 2.0
        self.turn_sign_correction = 1.0

        # Assignment-specific lap counting.
        # One lap of a rectangular/square course is treated as four right-angle turns.
        self.turn_count = 0
        self.lap_count = 0
        self.turn_history = []

        # LiDAR interpretation/classification output.
        self.map_observation_class = "unknown"
        self.front_clearance_m = np.inf
        self.left_clearance_m = np.inf
        self.right_clearance_m = np.inf

        # ---------------- Logging and ROS ----------------
        self.datalog = DataLogger(log_dir="logs")
        self.setup_experiment_csv()

        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd",
            Vector3Stamped,
            ip=self.robot_ip,
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",
            Vector3Stamped,
            self.true_wheel_speeds_callback,
            ip=self.robot_ip,
        )

        self.lidar_sub = Subscriber(
            "/lidar",
            RBLaserScan,
            self.lidar_callback,
            ip=self.robot_ip,
        )

        self.groundtruth_sub = Subscriber(
            "/groundtruth",
            PoseStamped,
            self.groundtruth_callback,
            ip=self.robot_ip,
        )

    # ------------------------------------------------------------------
    # Experiment logging
    # ------------------------------------------------------------------

    def setup_experiment_csv(self):
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_csv_path = log_dir / f"{stamp}_{self.experiment}_slam_metrics.csv"

        self.experiment_csv_file = self.experiment_csv_path.open("w", newline="", encoding="utf-8")
        self.experiment_writer = csv.writer(self.experiment_csv_file)

        self.experiment_writer.writerow([
            "time_s",
            "experiment",
            "n_particles",
            "process_v_std",
            "process_w_std_rad",
            "lidar_std",
            "map_std",
            "est_northing_m",
            "est_easting_m",
            "est_yaw_rad",
            "aruco_northing_m",
            "aruco_easting_m",
            "aruco_yaw_rad",
            "aruco_xy_std_m",
            "aruco_yaw_std_rad",
            "position_uncertainty_m",
            "front_clearance_m",
            "left_clearance_m",
            "right_clearance_m",
            "map_observation_class",
            "drive_state",
            "turn_count",
            "lap_count",
            "corner_northing_m",
            "corner_easting_m",
            "cmd_right_rad_s",
            "cmd_left_rad_s",
        ])

    def log_experiment_metrics(self):
        t_now = datetime.utcnow().timestamp()

        if len(self.corners) > 0:
            corner_n = self.corners[-1][0]
            corner_e = self.corners[-1][1]
        else:
            corner_n = None
            corner_e = None

        self.experiment_writer.writerow([
            t_now,
            self.experiment,
            self.n_particles,
            self.process_v_std,
            self.process_w_std,
            self.lidar_std,
            self.map_std,
            self.est_pose_northings_m,
            self.est_pose_eastings_m,
            self.est_pose_yaw_rad,
            self.measured_pose_northings_m,
            self.measured_pose_eastings_m,
            self.measured_pose_yaw_rad,
            ARUCO_XY_STD if ARUCO_NOISE_ENABLED else 0.0,
            np.deg2rad(ARUCO_YAW_STD_DEG) if ARUCO_NOISE_ENABLED else 0.0,
            self.position_uncertainty_m,
            self.front_clearance_m,
            self.left_clearance_m,
            self.right_clearance_m,
            self.map_observation_class,
            self.drive_state,
            self.turn_count,
            self.lap_count,
            corner_n,
            corner_e,
            self.cmd_wheelrate_right,
            self.cmd_wheelrate_left,
        ])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def add_noise_to_aruco_pose(self, n, e, yaw):
        if not ARUCO_NOISE_ENABLED:
            return float(n), float(e), wrap_angle(float(yaw))

        n_noisy = float(n) + self.aruco_rng.normal(0.0, ARUCO_XY_STD)
        e_noisy = float(e) + self.aruco_rng.normal(0.0, ARUCO_XY_STD)
        yaw_noisy = wrap_angle(
            float(yaw) + self.aruco_rng.normal(0.0, np.deg2rad(ARUCO_YAW_STD_DEG))
        )

        return n_noisy, e_noisy, yaw_noisy

    def detect_corner_gpr_style(self):
        if self.lidar_data_rb is None:
            return

        now = datetime.utcnow().timestamp()

        if now - self.last_corner_time < 1.5:
            return

        data = self.lidar_data_rb[~np.isnan(self.lidar_data_rb[:, 0])]

        if len(data) < 8:
            return

        ranges = data[:, 0]
        angles = data[:, 1]

        dr = np.abs(np.diff(ranges))
        corner_idx = int(np.argmax(dr))
        max_jump = float(dr[corner_idx])

        if max_jump < 0.20:
            return

        r_corner = float(ranges[corner_idx])
        a_corner = float(angles[corner_idx])

        if not np.isfinite(r_corner) or r_corner <= 0.0:
            return

        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m
        p_eb[1] = self.est_pose_eastings_m
        p_eb[2] = self.est_pose_yaw_rad

        z_lm = Vector(2)
        z_lm[0] = r_corner
        z_lm[1] = a_corner

        try:
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            n = float(t_em[0])
            e = float(t_em[1])

            if np.isfinite(n) and np.isfinite(e):
                self.corners.append([n, e])
                self.last_corner_time = now

                print(
                    "Detected LiDAR/GPR corner at n =",
                    round(n, 3),
                    "e =",
                    round(e, 3),
                    "range jump =",
                    round(max_jump, 3),
                )

        except Exception as ex:
            print("Corner conversion failed:", ex)

    def true_wheel_speeds_callback(self, msg):
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        print("Received lidar message", msg.header.seq)

        if self.sim_init:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        msg.header.stamp += self.sim_time_offset
        self.lidar_timestamp_s = msg.header.stamp

        ranges = np.array(msg.ranges, dtype=float)
        angles = np.array(msg.angles, dtype=float)

        self.lidar_data_rb = np.column_stack([
            np.where(ranges == 0.0, np.nan, ranges),
            angles,
        ])

        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m
        p_eb[1] = self.est_pose_eastings_m
        p_eb[2] = self.est_pose_yaw_rad

        self.lidar_data = np.full((len(ranges), 2), np.nan)
        z_lm = Vector(2)

        for i, r in enumerate(ranges):
            if r > 0.0:
                z_lm[0] = r
                z_lm[1] = angles[i]
                t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

                self.lidar_data[i, 0] = t_em[0]
                self.lidar_data[i, 1] = t_em[1]

        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

        self.update_map_observation_class()
        self.lidar_new = True
        self.datalog.log(msg, topic_name="/lidar")

    def groundtruth_callback(self, msg):
        self.datalog.log(msg, topic_name="/groundtruth")

    # ------------------------------------------------------------------
    # Pose parsing
    # ------------------------------------------------------------------

    def pose_parse(self, data, aruco=False):
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = data[0]

        pose_msg.pose.position.x = data[1]
        pose_msg.pose.position.y = data[2]
        pose_msg.pose.position.z = 0.0

        quat = Quaternion()

        if aruco and not self.simulation:
            quat.from_euler(0, 0, np.deg2rad(data[6]))
        else:
            quat.from_euler(0, 0, data[6])

        pose_msg.pose.orientation = quat
        return pose_msg

    # ------------------------------------------------------------------
    # Particle helpers
    # ------------------------------------------------------------------

    def initialise_particles(self):
        for p in self.particle_slam.particles:
            p.pose[0, 0] = self.est_pose_northings_m + np.random.normal(0.0, 0.03)
            p.pose[1, 0] = self.est_pose_eastings_m + np.random.normal(0.0, 0.03)
            p.pose[2, 0] = wrap_angle(
                self.est_pose_yaw_rad + np.random.normal(0.0, np.deg2rad(3.0))
            )
            p.weight = 1.0 / self.n_particles

    def get_odometry_input(self):
        if self.use_command_for_odometry:
            return self.last_u_cmd

        q_meas = Vector(2)

        q_meas[0] = (
            float(self.measured_wheelrate_right)
            if self.measured_wheelrate_right is not None
            else float(self.cmd_wheelrate_right)
        )

        q_meas[1] = (
            float(self.measured_wheelrate_left)
            if self.measured_wheelrate_left is not None
            else float(self.cmd_wheelrate_left)
        )

        return self.ddrive.fwd_kinematics(q_meas)

    def predict_estimate_and_particles(self):
        t_now = datetime.utcnow().timestamp()

        if self.t_prev is None:
            self.t_prev = t_now
            return

        dt = t_now - self.t_prev
        self.t_prev = t_now

        if dt <= 0.0:
            dt = 0.1
        if dt > 0.5:
            dt = 0.5

        u_odo = self.get_odometry_input()

        mu = Vector(3)
        mu[0] = self.est_pose_northings_m
        mu[1] = self.est_pose_eastings_m
        mu[2] = self.est_pose_yaw_rad

        mu = rigid_body_kinematics(mu, u_odo, dt)

        self.est_pose_northings_m = float(mu[0])
        self.est_pose_eastings_m = float(mu[1])
        self.est_pose_yaw_rad = wrap_angle(float(mu[2]))

        for p in self.particle_slam.particles:
            p_mu = Vector(3)
            p_mu[0] = float(p.pose[0, 0])
            p_mu[1] = float(p.pose[1, 0])
            p_mu[2] = float(p.pose[2, 0])

            u_noisy = Vector(2)
            u_noisy[0] = float(u_odo[0]) + np.random.normal(0.0, self.process_v_std)
            u_noisy[1] = float(u_odo[1]) + np.random.normal(0.0, self.process_w_std)

            p_mu = rigid_body_kinematics(p_mu, u_noisy, dt)

            p.pose[0, 0] = float(p_mu[0])
            p.pose[1, 0] = float(p_mu[1])
            p.pose[2, 0] = wrap_angle(float(p_mu[2]))

        self.update_uncertainty_from_particles()

    def update_uncertainty_from_particles(self):
        ns = np.array([float(p.pose[0, 0]) for p in self.particle_slam.particles])
        es = np.array([float(p.pose[1, 0]) for p in self.particle_slam.particles])

        self.particle_northings = ns.tolist()
        self.particle_eastings = es.tolist()

        if len(ns) > 1:
            self.position_uncertainty_m = float(np.sqrt(np.var(ns) + np.var(es)))
        else:
            self.position_uncertainty_m = 0.0

    def particle_path_slam_update(self):
        if self.lidar_data_rb is None:
            return

        valid = ~np.isnan(self.lidar_data_rb[:, 0])
        data = self.lidar_data_rb[valid]

        if len(data) == 0:
            return

        if len(data) > self.max_lidar_points_for_slam:
            idx = np.linspace(0, len(data) - 1, self.max_lidar_points_for_slam).astype(int)
            data = data[idx]

        # GPR-based map observation uncertainty model.
        # lidar_std scales measurement uncertainty with range.
        obs_std = data[:, [0]] * self.lidar_std
        obs_std = np.where(np.isnan(obs_std), self.lidar_std, obs_std)

        self.particle_slam.observation_update(
            data,
            obs_std,
            self.map_std,
            self.rbf_length_scale,
            resampling_flag=True,
            gpr_sample_cap=self.gpr_sample_cap,
            show_plot=False,
        )

        self.update_uncertainty_from_particles()

    # ------------------------------------------------------------------
    # LiDAR sector functions + map observation classification
    # ------------------------------------------------------------------

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

    def update_map_observation_class(self):
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
        self.update_map_observation_class()

        front_ranges = self.sector_ranges(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        if len(front_ranges) == 0:
            return False

        close_count = int(np.sum(front_ranges < self.front_block_threshold))
        return close_count >= self.min_front_close_beams

    def choose_turn_direction(self):
        self.update_map_observation_class()

        print("Left clearance:", self.left_clearance_m, "Right clearance:", self.right_clearance_m)
        print("Map observation class:", self.map_observation_class)

        if self.left_clearance_m >= self.right_clearance_m:
            return 1.0 * self.turn_sign_correction
        else:
            return -1.0 * self.turn_sign_correction

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def controller(self):
        wheel_msg = Vector3Stamped()
        u_cmd = Vector(2)

        now = datetime.utcnow().timestamp()

        if self.drive_state == "STRAIGHT":
            if now > self.turn_cooldown_until and self.front_blocked():
                self.turn_direction = self.choose_turn_direction()
                self.target_yaw_rad = wrap_angle(
                    self.est_pose_yaw_rad + self.turn_direction * np.pi / 2.0
                )
                self.turn_start_time = now
                self.drive_state = "TURNING"

                # Mark the detected corner/turn event for the visualiser.
                # This is cleaner than using raw range-jump corner detection.


                print("Obstacle ahead. Curving", "left" if self.turn_direction > 0 else "right")

                u_cmd[0] = self.v_turn
                u_cmd[1] = self.turn_direction * self.w_turn
            else:
                u_cmd[0] = self.v_forward
                u_cmd[1] = 0.0

        elif self.drive_state == "TURNING":
            err = angle_error(self.target_yaw_rad, self.est_pose_yaw_rad)

            if abs(err) < self.turn_tolerance or (now - self.turn_start_time) > self.max_turn_time_s:
                print("Finished curved turn. Going straight.")

                self.drive_state = "STRAIGHT"
                self.target_yaw_rad = None
                self.turn_start_time = None
                self.turn_cooldown_until = now + self.turn_cooldown_s

                self.turn_count += 1
                self.lap_count = self.turn_count // self.lap_corner
                self.turn_history.append([
                    now,
                    self.turn_count,
                    self.lap_count,
                    self.est_pose_northings_m,
                    self.est_pose_eastings_m,
                    self.est_pose_yaw_rad,
                    self.map_observation_class,
                ])

                print("Turn count:", self.turn_count, "Lap count:", self.lap_count)

                if self.turn_count >= self.lap_corner * self.required_laps:
                    print("Required laps completed. Stopping experiment.")
                    self.stop_flag = True
                    u_cmd[0] = 0.0
                    u_cmd[1] = 0.0
                else:
                    u_cmd[0] = self.v_forward
                    u_cmd[1] = 0.0
            else:
                u_cmd[0] = self.v_turn
                u_cmd[1] = self.turn_direction * self.w_turn

        else:
            u_cmd[0] = 0.0
            u_cmd[1] = 0.0

        self.last_u_cmd[0] = float(np.clip(u_cmd[0], -self.v_max, self.v_max))
        self.last_u_cmd[1] = float(np.clip(u_cmd[1], -self.w_max, self.w_max))

        q_cmd = self.ddrive.inv_kinematics(self.last_u_cmd)

        wheel_msg.vector.x = float(q_cmd[0, 0])
        wheel_msg.vector.y = float(q_cmd[1, 0])

        return wheel_msg

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def infinite_loop(self):
        aruco = self.aruco_driver.read()

        if aruco is not None:
            msg = self.pose_parse(aruco, aruco=True)

            self.measured_pose_timestamp_s = msg.header.stamp

            raw_n = float(msg.pose.position.x)
            raw_e = float(msg.pose.position.y)
            _, _, raw_yaw = msg.pose.orientation.to_euler()
            raw_yaw = wrap_angle(float(raw_yaw))

            self.raw_aruco_northings_m = raw_n
            self.raw_aruco_eastings_m = raw_e
            self.raw_aruco_yaw_rad = raw_yaw

            noisy_n, noisy_e, noisy_yaw = self.add_noise_to_aruco_pose(
                raw_n,
                raw_e,
                raw_yaw,
            )

            self.measured_pose_northings_m = noisy_n
            self.measured_pose_eastings_m = noisy_e
            self.measured_pose_yaw_rad = noisy_yaw

            # Log the original ArUco message unchanged. The noisy values are saved
            # in the experiment CSV as the ArUco measurement used by the algorithm.
            self.datalog.log(msg, topic_name="/aruco_raw")

            if not self.aruco_used_for_initial_pose:
                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = float(self.measured_pose_yaw_rad)

                self.initialise_particles()
                self.t_prev = datetime.utcnow().timestamp()
                self.aruco_used_for_initial_pose = True

                print("ArUco pose received and used for initial correction.")

        self.predict_estimate_and_particles()

        now = datetime.utcnow().timestamp()

        if self.lidar_new:
            self.detect_corner_gpr_style()

            if (now - self.last_slam_update_time) > self.slam_update_period_s:
                self.particle_path_slam_update()
                self.last_slam_update_time = now

            self.lidar_new = False

        self.northings_path.append(self.est_pose_northings_m)
        self.eastings_path.append(self.est_pose_eastings_m)

        est_msg = self.pose_parse([
            datetime.utcnow().timestamp(),
            self.est_pose_northings_m,
            self.est_pose_eastings_m,
            0.0,
            0.0,
            0.0,
            self.est_pose_yaw_rad,
        ])

        self.datalog.log(est_msg, topic_name="/est_pose")

        wheel_msg = self.controller()

        self.cmd_wheelrate_right = wheel_msg.vector.x
        self.cmd_wheelrate_left = wheel_msg.vector.y

        print(
            "CMD wheel speeds:",
            round(self.cmd_wheelrate_right, 3),
            round(self.cmd_wheelrate_left, 3),
            "| uncertainty:",
            round(self.position_uncertainty_m, 3),
            "| class:",
            self.map_observation_class,
        )

        if not self.stop_flag:
            self.wheel_speed_pub.publish(wheel_msg)
        else:
            stop_msg = Vector3Stamped()
            stop_msg.vector.x = 0.0
            stop_msg.vector.y = 0.0
            self.wheel_speed_pub.publish(stop_msg)

        self.datalog.log(wheel_msg, topic_name="/wheel_speeds_cmd")
        self.log_experiment_metrics()

    # ------------------------------------------------------------------
    # Run/stop
    # ------------------------------------------------------------------

    def stopcommand(self):
        print("Wheels stopping")

        self.stop_flag = True
        r = Rate(10)

        stop_msg = Vector3Stamped()
        stop_msg.vector.x = 0.0
        stop_msg.vector.y = 0.0

        for _ in range(10):
            self.wheel_speed_pub.publish(stop_msg)
            r.sleep()

        self.lidar_sub.stop()
        self.true_wheel_speed_sub.stop()
        self.groundtruth_sub.stop()

        try:
            self.experiment_csv_file.flush()
            self.experiment_csv_file.close()
            print("Experiment metrics saved in", self.experiment_csv_path)
        except Exception:
            pass

        print("Data saved in", self.datalog.filename)

    def run(self, time_to_run=-1):
        self.start_time = datetime.utcnow().timestamp()

        try:
            r = Rate(10)

            while True:
                now = datetime.utcnow().timestamp()

                if time_to_run > 0 and now - self.start_time > time_to_run:
                    print("Time is up, stopping.")
                    break

                if self.stop_flag:
                    print("Stop flag set. Ending run.")
                    break

                self.infinite_loop()
                r.sleep()

        except KeyboardInterrupt:
            print("KeyboardInterrupt received.")

        except Exception as e:
            print("Exception:", e)

        finally:
            self.stopcommand()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--physical",
        action="store_true",
        help="Use physical robot. Default is Webots simulation.",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Run time in seconds. Negative means run forever.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=list(EXPERIMENT_PRESETS.keys()),
        help="Experiment preset for process noise, map observation noise, and particle count.",
    )

    parser.add_argument(
        "--laps",
        type=int,
        default=2,
        help="Number of rectangular laps to complete. One lap is counted as four right-angle turns.",
    )

    args = parser.parse_args()

    bot = LaptopPilot(
        simulation=not args.physical,
        experiment=args.experiment,
        laps=args.laps,
    )
    bot.run(args.time)
