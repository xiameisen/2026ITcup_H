import time

from ybUtils.YbUart import YbUart


uart = YbUart(baudrate=115200)

while True:
    uart.send("$B,+012,-034,095,1\n")
    time.sleep(0.1)
