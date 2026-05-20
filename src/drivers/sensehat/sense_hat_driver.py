# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import weakref
import argparse

from drivers.rpi import Console

from .sense_hat_controller import SenseHatController
print('driving')

class RGBColor:
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    PURPLE = (160, 32, 240)
    WHITE = (255, 255, 255)


class PixelMatrix:
    def __init__(self):
        self._mat = [(0, 0, 0) for _ in range(64)]

    def set_pixel(self, x, y, color):
        self._mat[x + 8 * y] = color

    def get_pixel(self, x, y):
        return self._mat[x + 8 * y]

    def get(self):
        # Returns a list of 64 smaller lists of [R,G,B] pixels
        return self._mat


class SenseHatDriver:
    def __init__(self, parent=None):
        if parent is not None:
            self._parent = weakref.ref(parent)
        self.ready = False
        try:
            self._dev = SenseHatController()
            self.ready = True
        except Exception as e:
            Console.warn("    SenseHatCompass could not be initialised. Error:", e)

    def read(self) -> tuple[float, float, float]:
        if not self.ready:
            Console.warn("    SenseHatCompass is not ready")
            return None, None, None
        angular_velocity = self._dev.get_gyroscope_raw()
        xs = angular_velocity["x"] 
        ys = angular_velocity["y"]
        zs = angular_velocity["z"]
        return xs, ys, zs

    def print(self, msg):
        """
        Scrolls a string of text across the LED matrix.
        """
        self._dev.show_message(msg)

    def print_pixel_list(self, pixel_list):
        """
        Accepts a list containing 64 smaller lists of [R,G,B] pixels and
        updates the LED matrix. R,G,B elements must intergers between 0
        and 255.
        """
        self._dev.set_pixels(pixel_list)
