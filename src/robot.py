# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
import time
import os

import numpy as np
from zeroros import MessageBroker, Publisher, Subscriber
from zeroros.messages import String, Vector3

from drivers.thrusters import ThrustersDriver
from drivers.sensehat.sense_hat_driver import (
    SenseHatDriver,
    PixelMatrix,
    RGBColor,
)
from drivers.sonar.echosounder_driver import Echosounder
from drivers.rpi import Console, Rate

"""
WARNING - This is the code that runs on the robot.
It is not meant to be run on the laptop.

DO NOT MODIFY THIS FILE.

It will have no effect on the robot.
"""

class Robot:
    def __init__(self):
        Console.set_logging_file("/home/robot/logs", name="robot")
        self.broker = MessageBroker(ip="*")
        self.configured = False
        self.config_sub = Subscriber("/config", String, self.config_cb)
        self.shutdown_sub = Subscriber("/shutdown", String, self.shutdown_cb)
        self.reboot_sub = Subscriber("/reboot", String, self.reboot_cb)
        self.console_pub = Publisher("/command", String)
        self.rate = 5.0
        self.sense_hat = SenseHatDriver()
        self.sense_hat_matrix = PixelMatrix()
        self.echosounder = Echosounder()
        self.echosounder.ping_echosounder()
        self.thrusters = None
        self.count = 0
        self.stop = False
        while not self.configured and not self.stop:
            try:
                Console.info("Waiting for configuration on topic /config ...")
                if self.count % 2 == 0:
                    self.sense_hat_matrix.set_pixel(3, 3, RGBColor.RED)
                    self.sense_hat_matrix.set_pixel(4, 3, RGBColor.RED)
                    self.sense_hat_matrix.set_pixel(3, 4, RGBColor.RED)
                    self.sense_hat_matrix.set_pixel(4, 4, RGBColor.RED)
                    self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())
                else:
                    self.sense_hat_matrix.set_pixel(3, 3, RGBColor.WHITE)
                    self.sense_hat_matrix.set_pixel(4, 3, RGBColor.WHITE)
                    self.sense_hat_matrix.set_pixel(3, 4, RGBColor.WHITE)
                    self.sense_hat_matrix.set_pixel(4, 4, RGBColor.WHITE)
                    self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())
                time.sleep(1.0)
                self.count += 1
            except KeyboardInterrupt:
                Console.info("Ctrl+C pressed. Stopping...")
                self.config_sub.stop()
                self.shutdown_sub.stop()
                self.broker.stop()
                return

        # Reset LEDs
        self.sense_hat_matrix = PixelMatrix()
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())

        Console.info("Starting robot...")
        self.console_pub.publish(String("Starting robot..."))
        r = Rate(self.rate)

        # Reset thrusters
        for i in range(10):
            self.thrusters.move(0,0)
            r.sleep()

        while not self.stop:
            try:
                Console.info("Looping")
                self.loop()
            except KeyboardInterrupt:
                Console.info("Ctrl+C pressed. Stopping...")
                break
            r.sleep()
        self.config_sub.stop()
        self.shutdown_sub.stop()
        self.broker.stop()

    def __del__(self):
        if self.thrusters is not None:
            self.thrusters.move(0, 0)
        self.sense_hat_matrix = PixelMatrix()
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())

    def config_cb(self, msg: String):
        self.thrusters = ThrustersDriver(parent=self)
        self.imu_pub = Publisher("/imu", Vector3)
        self.sonar_pub = Publisher("/sonar", Vector3)
        self.prop_rate_sub = Subscriber("/control",Vector3, self.cmd_rate_cb)
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())
        self.configured = True

    def loop(self):
        X, Y, Z = None, None, None
        if not self.configured:
            return
        X, Y, Z = self.sense_hat.read()
        self.imu_pub.publish(Vector3(X, Y, Z))
        echorange, echoconfidence = self.echosounder.sonar()
        self.sonar_pub.publish(Vector3(0,0,echorange))
        Console.info("Sent sensor data:", X, Y, Z, echorange)
        self.sense_hat_matrix.set_pixel(self.count % 8, 0, RGBColor.BLUE)
        self.sense_hat_matrix.set_pixel((self.count + 1) % 8, 0, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 2) % 8, 0, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 3) % 8, 0, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 4) % 8, 0, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 5) % 8, 0, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 6) % 8, 0, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 7) % 8, 0, RGBColor.BLACK)
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())

    def cmd_rate_cb(self, msg: Vector3):
        Console.info("Received cmd_rate message (right_rate, left_rate):", msg.x, msg.y)
        self.thrusters.move(msg.x, msg.y)
        self.sense_hat_matrix.set_pixel(self.count % 8, 1, RGBColor.GREEN)
        self.sense_hat_matrix.set_pixel((self.count + 1) % 8, 1, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 2) % 8, 1, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 3) % 8, 1, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 4) % 8, 1, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 5) % 8, 1, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 6) % 8, 1, RGBColor.BLACK)
        self.sense_hat_matrix.set_pixel((self.count + 7) % 8, 1, RGBColor.BLACK)
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())
        self.count += 1

    def shutdown_cb(self, msg: String):
        self.configured = False
        self.sense_hat.print("Shutdown...")
        self.sense_hat_matrix = PixelMatrix()
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())
        os.system("sudo shutdown -h now")
        self.stop = True

    def reboot_cb(self, msg: String):
        self.configured = False
        self.sense_hat.print("Reboot...")
        self.sense_hat_matrix = PixelMatrix()
        self.sense_hat.print_pixel_list(self.sense_hat_matrix.get())
        os.system("sudo reboot")
        self.stop = True

def main():
    r = Robot()
    del r

if __name__ == "__main__":
    main()
