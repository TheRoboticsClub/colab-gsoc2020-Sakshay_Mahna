#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Alberto Martin Florido <almartinflorido@gmail.com>
#       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
#
import threading
import time
import rospy
from datetime import datetime

time_cycle = 20


class ThreadPublisher(threading.Thread):

    def __init__(self, pub, kill_event, clock):
        self.pub = pub
        self.kill_event = kill_event
        threading.Thread.__init__(self, args=kill_event)
        self.clock = clock

    def run(self):
        while (not self.kill_event.is_set()):
            start_time = self.clock.getTimeData()

            try:
                self.pub.publish()
            except rospy.ROSException:
                pass

            finish_Time = self.clock.getTimeData()

            dt = finish_Time - start_time
            ms = dt
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)
