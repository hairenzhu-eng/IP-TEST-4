from . import Ping1D
import time

class Echosounder:
    def __init__(self, parent=None):
        self.ready = False
        self.myPing = Ping1D()
        self.myPing.connect_serial("/dev/ttyUSB0", 115200)

    def ping_echosounder(self):
        while self.myPing.initialize() is False:
            time.sleep(1.0)
        self.ready = True

    def sonar(self):
        sonardata = self.myPing.get_distance()
        if sonardata != None:
            self.depth = sonardata["distance"]
            self.depth_confidence = sonardata["confidence"]
        else:
            self.depth = None
            self.depth_confidence = None
        return(self.depth, self.depth_confidence)
