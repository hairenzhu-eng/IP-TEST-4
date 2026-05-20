import time

import zeroros
from zeroros.messages import String

robot_ip = "192.168.10.1"
reboot_pub = zeroros.Publisher("/reboot", String, ip=robot_ip)
for i in range(60):
    reboot_pub.publish(String(""))
    print("Rebooting. Wait 30s")    
    time.sleep(0.5)
