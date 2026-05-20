import argparse

from .sense_hat import SenseHat


parser = argparse.ArgumentParser()
parser.add_argument(
    "msg",
    type=str,
    help="Message to be printed on the LED matrix",
)

sense = SenseHat()
sense.show_message(parser.parse_args().msg)
