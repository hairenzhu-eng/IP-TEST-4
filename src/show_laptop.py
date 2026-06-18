"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import time
import sys
from threading import Thread
from time import sleep
import argparse
import numpy as np
from drivers.rpi import Rate

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot import LiveScatterPlot
from pglive.sources.live_plot_widget import LivePlotWidget
import laptop as lt

class ShowLaptop(QWidget):
    obstacle_summary_signal = pyqtSignal(str)
    running = False

    def __init__(self, parent=None):

        rate = 5.0
        self.r = Rate(rate)
        self.lastdt = 1/rate

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.rpmplot = LivePlotWidget()
        self.headingplot = LivePlotWidget()
        self.positionplot = LivePlotWidget()
        self.timeplot = LivePlotWidget()
        self.depthplot = LivePlotWidget()
        layout = QGridLayout(self)
        layout.addWidget(self.rpmplot, 0, 3, 1, 2)
        layout.addWidget(self.headingplot, 1, 3, 1, 2)
        layout.addWidget(self.positionplot, 0, 0, 2, 2)
        layout.addWidget(self.timeplot, 2, 0, 1, 2)
        layout.addWidget(self.depthplot, 2, 3, 1, 2)
        self.loopcounter = 0
        self.Laptop = lt.LaptopController(
            OPERATING_MODE,
            obstacle_ekf_prediction_enabled=OBSTACLE_EKF_PREDICTION_ENABLED,
        )
        self.obstacle_summary_signal.connect(self._set_obstacle_summary_title)
        
        # Create one curve pre dataset
        thruster1plot = LiveLinePlot(pen="blue", name = 'Thruster 1')
        thruster2plot = LiveLinePlot(pen="red", name = 'Thruster 2')
        
        EKFheadingplot = LiveLinePlot(pen = 'blue', name = 'Model Heading')
        sensedheadingplot = LiveLinePlot(pen='red', name = 'IMU Integrated Heading')
        ARUCOheadingplot = LiveLinePlot(symbol = 'x', pen = 'green', name = 'ARUCO Sensed Heading')       
        
        positionplot = LiveLinePlot(pen = 'blue', name = 'Model Path')
        ARUCOplot = LiveScatterPlot(symbol = 'x', pen = 'green', name = 'ARUCO Sensed Position')
        WayPoint = LiveScatterPlot(symbol = 'o', pen = 'red', name = 'Waypoints')
        lidarplot = LiveScatterPlot(symbol = 'o', size = 1, pen = 'w', name = 'Lidar')
        plannedPathPlot = LiveLinePlot(pen = 'gray', name = 'Planned Path')
        apfForcePlot = LiveLinePlot(pen = 'yellow', name = 'APF Total Force')
        apfRepulsivePlot = LiveLinePlot(pen = 'magenta', name = 'APF Repulsive Force')
        apfSteeringPlot = LiveLinePlot(pen = 'cyan', name = 'APF Steering Force')
        apfTargetPlot = LiveScatterPlot(symbol = 't', size = 10, pen = 'orange', name = 'APF Target')
        obstacleEkfPositionPlot = LiveScatterPlot(symbol = 'o', size = 8, pen = 'orange', name = 'Obstacle EKF Position')
        obstacleEkfHistoryPlot = LiveLinePlot(pen = 'green', name = 'Obstacle EKF Track')
        obstacleEkfPredictionPlot = LiveLinePlot(pen = 'orange', name = 'Obstacle EKF Prediction')
        obstacleEkfDirectionPlot = LiveLinePlot(pen = 'red', name = 'Obstacle Direction')
        
        dtplot = LiveLinePlot(pen='blue', name = 'Laptop Update')
        Idtplot = LiveScatterPlot(symbol = 'x', pen = 'red', name = 'IMU Update')
        Adtplot = LiveScatterPlot(symbol = 'x', pen = 'green', name = 'ARUCO Update')

        depthplot = LiveLinePlot(pen = 'red', name = 'Sensed Depth')

        # Data connectors for each plot with dequeue of 600 points
        self.thruster1plot = DataConnector(thruster1plot, max_points=1500)
        self.thruster2plot = DataConnector(thruster2plot, max_points=1500)
        
        self.DHP = DataConnector(ARUCOheadingplot, max_points=1500)
        self.SHP = DataConnector(sensedheadingplot, max_points=1500)
        self.EHP = DataConnector(EKFheadingplot, max_points=1500)
        
        self.pos = DataConnector(positionplot, max_points=1500)
        self.ASP = DataConnector(ARUCOplot, max_points=1500)
        self.WP = DataConnector(WayPoint, max_points=1500)
        self.lidar = DataConnector(lidarplot, max_points=3000)
        self.planned_path = DataConnector(plannedPathPlot, max_points=20)
        self.apf_force = DataConnector(apfForcePlot, max_points=2)
        self.apf_repulsive = DataConnector(apfRepulsivePlot, max_points=2)
        self.apf_steering = DataConnector(apfSteeringPlot, max_points=2)
        self.apf_target = DataConnector(apfTargetPlot, max_points=1)
        self.obstacle_ekf_position = DataConnector(obstacleEkfPositionPlot, max_points=50)
        self.obstacle_ekf_history = DataConnector(obstacleEkfHistoryPlot, max_points=3000)
        self.obstacle_ekf_prediction = DataConnector(obstacleEkfPredictionPlot, max_points=1000)
        self.obstacle_ekf_direction = DataConnector(obstacleEkfDirectionPlot, max_points=200)
        
        self.dtplot = DataConnector(dtplot, max_points=1500)
        self.Idtplot = DataConnector(Idtplot, max_points=1500)
        self.Adtplot = DataConnector(Adtplot , max_points=1500)

        self.deplot = DataConnector(depthplot, max_points=1500)

        # Create plot itself
        #self.rpmplot = LivePlotWidget(title="Line Plot - Time series @ 2Hz", axisItems={'bottom': bottom_axis})
        # Show grid
        self.rpmplot.showGrid(x=True, y=True, alpha=0.3)
        self.headingplot.showGrid(x=True, y=True, alpha=0.3)
        self.positionplot.setAspectLocked()
        self.positionplot.showGrid(x = True, y = True, alpha = 0.3)
        self.timeplot.showGrid(x = True, y = True, alpha = 0.3)
        self.depthplot.showGrid(x = True, y = True, alpha = 0.3)

        # Set labels
        self.rpmplot.setLabel('bottom', 'Time', units="s")
        self.rpmplot.setLabel('left', 'Thruster RPM')
        self.rpmplot.addLegend()
        self.headingplot.setLabel('bottom', 'Time', units="s")
        self.headingplot.setLabel('left', 'Heading', units="degrees")
        self.headingplot.addLegend()
        self.positionplot.setLabel('bottom', 'East', units="m")
        self.positionplot.setLabel('left', 'North', units="m")
        self.positionplot.addLegend()
        self.timeplot.setLabel('bottom', 'Time', units="s")
        self.timeplot.setLabel('left', 'Time from last update', units="s")
        self.timeplot.addLegend()
        self.depthplot.setLabel('bottom', 'Time', units="s")
        self.depthplot.setLabel('left', 'Depth of bottom', units="m")
        self.depthplot.addLegend()
        # Add all three curves
        self.rpmplot.addItem(thruster1plot)
        self.rpmplot.addItem(thruster2plot)
        self.headingplot.addItem(ARUCOheadingplot)
        self.headingplot.addItem(sensedheadingplot)
        self.headingplot.addItem(EKFheadingplot)
        self.positionplot.addItem(positionplot)
        self.positionplot.addItem(ARUCOplot)
        self.positionplot.addItem(WayPoint)
        self.positionplot.addItem(plannedPathPlot)
        self.positionplot.addItem(lidarplot)
        self.positionplot.addItem(apfForcePlot)
        self.positionplot.addItem(apfRepulsivePlot)
        self.positionplot.addItem(apfSteeringPlot)
        self.positionplot.addItem(apfTargetPlot)
        self.positionplot.addItem(obstacleEkfPositionPlot)
        self.positionplot.addItem(obstacleEkfHistoryPlot)
        self.positionplot.addItem(obstacleEkfPredictionPlot)
        self.positionplot.addItem(obstacleEkfDirectionPlot)
        self.timeplot.addItem(dtplot)
        self.timeplot.addItem(Idtplot)
        self.timeplot.addItem(Adtplot)
        self.depthplot.addItem(depthplot)

        # using -1 to span through all rows available in the window
        #layout.addWidget(self.rpmplot, 2, 0, -1, 3)
        
        self.ST = timestamp = time.time()   
        self.lastimu = 0
        self.lastARUCO = 0
        self.lastdepth = 0
        self._lidar_timestamp_s_prev = None
        self._map_x = []
        self._map_y = []
        self._map_x_store = []
        self._map_y_store = []

    def _set_obstacle_summary_title(self, summary):
        self.positionplot.setTitle(summary)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            enabled = self.Laptop.toggle_obstacle_ekf_prediction()
            state = "ON" if enabled else "OFF"
            self.obstacle_summary_signal.emit(f"Obstacle EKF prediction: {state}")
            event.accept()
            return

        super().keyPressEvent(event)
        
        
    def _force_line_ne(self, force_body, scale=0.8):
        if force_body is None or self.Laptop.North is None or self.Laptop.East is None:
            return None

        force_body = np.asarray(force_body, dtype=float).reshape(2)
        if not np.isfinite(force_body).all():
            return None

        force_norm = float(np.linalg.norm(force_body))
        if force_norm < 1e-6:
            return None

        origin_ne = np.array([float(self.Laptop.North), float(self.Laptop.East)], dtype=float)
        force_ne = self.Laptop.body_vector_to_earth(force_body)
        force_ne_norm = float(np.linalg.norm(force_ne))
        if force_ne_norm < 1e-6 or not np.isfinite(force_ne_norm):
            return None

        line_len_m = min(1.2, max(0.25, scale * force_norm))
        end_ne = origin_ne + force_ne / force_ne_norm * line_len_m
        return [origin_ne[0], end_ne[0]], [origin_ne[1], end_ne[1]]

    def _set_force_line(self, connector, force_body, scale=0.8):
        line = self._force_line_ne(force_body, scale)
        if line is None:
            connector.cb_set_data([], [])
            return

        northings, eastings = line
        connector.cb_set_data(northings, eastings)

    def _update_apf_plot(self):
        self._set_force_line(self.apf_force, getattr(self.Laptop, "apf_force_body", None), scale=0.8)
        self._set_force_line(self.apf_repulsive, getattr(self.Laptop, "apf_repulsive_force_body", None), scale=0.8)
        self._set_force_line(self.apf_steering, getattr(self.Laptop, "apf_steering_force_body", None), scale=0.8)

        target_ne = np.asarray(
            getattr(self.Laptop, "apf_target_ne", [np.nan, np.nan]),
            dtype=float,
        ).reshape(2)
        if np.isfinite(target_ne).all():
            self.apf_target.cb_set_data([target_ne[0]], [target_ne[1]])
        else:
            self.apf_target.cb_set_data([], [])

    def _append_track_segments(self, northings, eastings, points_ne):
        points_ne = np.asarray(points_ne, dtype=float)
        if points_ne.ndim != 2 or points_ne.shape[0] < 2 or points_ne.shape[1] < 2:
            return

        finite = np.isfinite(points_ne[:, 0]) & np.isfinite(points_ne[:, 1])
        points_ne = points_ne[finite]
        if len(points_ne) < 2:
            return

        northings.extend(points_ne[:, 0].tolist())
        eastings.extend(points_ne[:, 1].tolist())
        northings.append(np.nan)
        eastings.append(np.nan)

    def _update_obstacle_tracks_plot(self):
        tracks = self.Laptop.obstacle_track_visuals()
        position_northings = []
        position_eastings = []
        history_northings = []
        history_eastings = []
        prediction_northings = []
        prediction_eastings = []
        direction_northings = []
        direction_eastings = []
        summaries = []

        for track in tracks:
            position_ne = np.asarray(track.get("position_ne", [np.nan, np.nan]), dtype=float).reshape(2)
            velocity_ne = np.asarray(track.get("velocity_ne", [0.0, 0.0]), dtype=float).reshape(2)

            if not np.isfinite(position_ne).all():
                continue

            position_northings.append(position_ne[0])
            position_eastings.append(position_ne[1])
            self._append_track_segments(history_northings, history_eastings, track.get("history_ne", []))
            self._append_track_segments(prediction_northings, prediction_eastings, track.get("prediction_ne", []))

            heading_deg = float(track.get("heading_deg", np.nan))
            if np.isfinite(heading_deg):
                summaries.append(f"#{int(track.get('id', 0))} {heading_deg:.0f}deg")

            speed_m_s = float(track.get("speed_m_s", np.linalg.norm(velocity_ne)))
            if np.isfinite(speed_m_s) and speed_m_s >= 1e-3 and np.isfinite(velocity_ne).all():
                vector_len_m = 0.45
                direction_ne = velocity_ne / max(float(np.linalg.norm(velocity_ne)), 1e-6)
                end_ne = position_ne + direction_ne * vector_len_m
                direction_northings.extend([position_ne[0], end_ne[0], np.nan])
                direction_eastings.extend([position_ne[1], end_ne[1], np.nan])

        self.obstacle_ekf_position.cb_set_data(position_northings, position_eastings)
        self.obstacle_ekf_history.cb_set_data(history_northings, history_eastings)
        self.obstacle_ekf_prediction.cb_set_data(prediction_northings, prediction_eastings)
        self.obstacle_ekf_direction.cb_set_data(direction_northings, direction_eastings)
        self.obstacle_summary_signal.emit("Obstacle EKF: " + " | ".join(summaries[:3]) if summaries else "")

    def _update_lidar_plot(self, lidar_cloud_ne):
        lidar_timestamp_s = getattr(self.Laptop, "latest_lidar_received_s", None)
        if lidar_cloud_ne is None or lidar_timestamp_s == self._lidar_timestamp_s_prev:
            return

        valid_northings = []
        valid_eastings = []
        for point in lidar_cloud_ne:
            if len(point) < 2:
                continue
            if not np.isnan(point[0]) and not np.isnan(point[1]):
                self._map_x_store.append(point[0])
                self._map_y_store.append(point[1])
                self._map_x.append(point[0])
                self._map_y.append(point[1])
                valid_northings.append(point[0])
                valid_eastings.append(point[1])

        if valid_northings:
            self.lidar.cb_append_data_array(valid_northings, valid_eastings)

        if len(self._map_x) > 1500 and len(self._map_x_store) >= 1000:
            ind = np.random.choice(len(self._map_x_store), 1000, replace=False)
            self.lidar.cb_set_data(
                [self._map_x_store[i] for i in ind],
                [self._map_y_store[i] for i in ind],
            )
            self._map_x = []
            self._map_y = []

        self._lidar_timestamp_s_prev = lidar_timestamp_s


    def update(self):
        """Generate data at 2Hz"""
        while self.running:
                    
            right_rate, left_rate, lastdt, current_heading, North, East, sensed_yaw_rate, sensed_yaw, imu_time1, sensed_pos_northings_m, sensed_pos_eastings_m, sensed_pos_yaw_rad, ARUCO_time1, Waypoints, reference_path, depth, depth_time1, mission_complete, lidar_cloud_ne = self.Laptop.loop()
            if self.loopcounter == 0 and Waypoints != None:
                for i in range(len(Waypoints)):
                    self.WP.cb_append_data_point(Waypoints[i].y, Waypoints[i].x)
                self.planned_path.cb_set_data(
                    [waypoint.y for waypoint in Waypoints],
                    [waypoint.x for waypoint in Waypoints],
                )
            self.loopcounter = self.loopcounter + 1            
            self.TFS = time.time() - self.ST
            
            if left_rate != None and right_rate != None:
                self.thruster1plot.cb_append_data_point(left_rate*60/(2*np.pi), self.TFS)
                self.thruster2plot.cb_append_data_point(right_rate*60/(2*np.pi), self.TFS)
            
            if imu_time1 != None:
                imu_time = imu_time1 - self.ST
            else:
                imu_time = None
                
            if ARUCO_time1 != None:
                ARUCO_time = ARUCO_time1 - self.ST
            else:
                ARUCO_time = None

            if depth_time1 != None:
                depth_time = depth_time1 - self.ST
            else:
                depth_time = None                
            
            if sensed_yaw != None and imu_time >= self.TFS - lastdt:
                self.SHP.cb_append_data_point(np.rad2deg(sensed_yaw), self.TFS)
            if sensed_pos_yaw_rad != None:
                self.DHP.cb_append_data_point(np.rad2deg(sensed_pos_yaw_rad), self.TFS)
            if current_heading != None:
                self.EHP.cb_append_data_point(np.rad2deg(current_heading), self.TFS)
            
            if East != None and North != None:
                self.pos.cb_append_data_point(North,East)
            if sensed_pos_northings_m != None and sensed_pos_eastings_m != None:
                self.ASP.cb_append_data_point(sensed_pos_northings_m, sensed_pos_eastings_m)
            self._update_lidar_plot(lidar_cloud_ne)
            self._update_apf_plot()
            self._update_obstacle_tracks_plot()
            
            if lastdt != None:
                self.dtplot.cb_append_data_point(lastdt, self.TFS)
                
            if imu_time != None and imu_time >= self.TFS - lastdt:
                self.Idtplot.cb_append_data_point(imu_time - self.lastimu, imu_time)
            if ARUCO_time != None:
                self.Adtplot.cb_append_data_point(ARUCO_time - self.lastARUCO, ARUCO_time)  

            if depth_time != None and depth_time >= self.TFS - lastdt:
                self.deplot.cb_append_data_point(depth, self.TFS)
                
            if imu_time != None and imu_time >= self.TFS - lastdt:
                self.lastimu = imu_time
            if ARUCO_time != None:
                self.lastARUCO = ARUCO_time 
            if depth_time != None and depth_time >= self.TFS - lastdt:
                self.lastdepth = depth_time               
            
            self.ET = timestamp = time.time()

            self.r.sleep()
            
    def breaker(self):
        self.Laptop.stopcommand()
        

    def start_app(self):
        """Start Thread generator"""
        self.running = True
        Thread(target=self.update).start()
        
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )
    lt.add_obstacle_ekf_prediction_args(parser)

    args = parser.parse_args()

    if args.simulation == True: 
        OPERATING_MODE = 2
        print('Running laptop.py in simulation')
    else: 
        OPERATING_MODE = 1
        print('Running laptop.py on robot')

    OBSTACLE_EKF_PREDICTION_ENABLED = args.obstacle_ekf_prediction_enabled
    print(
        "Obstacle EKF prediction:",
        "enabled" if OBSTACLE_EKF_PREDICTION_ENABLED else "disabled",
    )



    app = QApplication(sys.argv)
    window = ShowLaptop()
    window.show()
    window.start_app()
    app.exec()
    window.running = False
    window.breaker()
