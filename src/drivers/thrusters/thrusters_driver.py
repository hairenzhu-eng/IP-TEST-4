# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import time
import weakref

import numpy as np

from drivers.rpi import Console

from .maestro_controller import MaestroController

max_fwd = 8000
max_rev = 3200
neutral = 5600
deadband = 500

def rad2pwm(rad):
    scale_fwd = (max_fwd - (neutral + deadband)) / (200)
    scale_rev = ((neutral - deadband) - max_rev) / (200)

    if rad > 0:
        return (rad) * scale_fwd + (neutral + deadband)
    elif rad < 0:
        return (rad ) * scale_rev + (neutral - deadband)
    else:
        # snap to boundary
        return neutral

class ThrustersDriver:
    def __init__(self, parent=None):
        self.ready = True
        self.name = "pololu_motors"
        if parent is not None:
            self._parent = weakref.ref(parent)

        self._ctrl = MaestroController()

        self.left_idx = 0
        self.right_idx = 1

        self._ctrl.setRange(self.left_idx, max_rev, max_fwd)
        self._ctrl.setRange(self.right_idx, max_rev, max_fwd)

        self.current_right_rate = neutral
        self.current_left_rate = neutral

    def init(self):
        try:
            # For ESC to boot, we need to stop at neutral
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(7.0)
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)

            self.current_right_rate = neutral
            self.current_left_rate = neutral

            self.ready = True
        except Exception as e:
            Console.warn("    PololuMotors could not be initialised. Error:", e)
            self.ready = False

        return self.ready

    def move(self, right_rate: float, left_rate: float):
        if not self.ready:
            Console.warn("    PololuMicroMaestro is not ready", self.ready)
            return False

        target_right_rate = rad2pwm(right_rate)
        target_left_rate= rad2pwm(left_rate)

        # --- handle direction change right side ---
        if ((self.current_right_rate < neutral < target_right_rate) or (self.current_right_rate > neutral > target_right_rate)) and self.current_right_rate != neutral:
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.1)

        # --- handle direction change left side ---
        if ((self.current_left_rate < neutral < target_left_rate) or (self.current_left_rate > neutral > target_left_rate)) and self.current_left_rate != neutral:
            self._ctrl.setTarget(self.left_idx, neutral)
            time.sleep(0.1)

        self.current_right_rate = target_right_rate
        self.current_left_rate = target_left_rate

        self._ctrl.setTarget(self.right_idx, int(self.current_right_rate))
        self._ctrl.setTarget(self.left_idx, int(self.current_left_rate))

        return True

    def __del__(self):
        if self.ready:
            Console.info("Stopping the motors...")
            # For ESC to boot, we need to stop at neutral
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)
            self._ctrl.setTarget(self.left_idx, neutral)
            self._ctrl.setTarget(self.right_idx, neutral)
            time.sleep(0.5)
            self._ctrl.close()
