import argparse
from .sense_hat import SenseHat

parser = argparse.ArgumentParser()
parser.add_argument(
    "pixel_x",
    type=int,
    default=0,
    help="X coordinate of the pixel to be set",
)

parser.add_argument(
    "pixel_y",
    type=int,
    default=0,
    help="Y coordinate of the pixel to be set",
)

parser.add_argument(
    "color",
    type=str,
    default="blue",
    help="Color of the pixel to be set",
)


sense = SenseHat()

red = (255, 0, 0)
orange = (255, 165, 0)
yellow = (255, 255, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
purple = (160, 32, 240)
white = (255, 255, 255)

args = parser.parse_args()

color = args.color
if color == "red":
    sense.set_pixel(args.pixel_x, args.pixel_y, red)
elif color == "orange":
    sense.set_pixel(args.pixel_x, args.pixel_y, orange)
elif color == "yellow":
    sense.set_pixel(args.pixel_x, args.pixel_y, yellow)
elif color == "green":
    sense.set_pixel(args.pixel_x, args.pixel_y, green)
elif color == "blue":
    sense.set_pixel(args.pixel_x, args.pixel_y, blue)
elif color == "purple":
    sense.set_pixel(args.pixel_x, args.pixel_y, purple)
else:
    sense.set_pixel(args.pixel_x, args.pixel_y, white)
