#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import subprocess

THERMAL_PATH = '/sys/devices/virtual/thermal/'


def read_sys_value(pth):
    return subprocess.check_output(['cat', pth]).decode('utf-8').rstrip('\n')


class JetsonParams():
    def __init__(self):
        '''
        Returns a list of the thermal zone paths
        '''
        self.zone_paths = [os.path.join(THERMAL_PATH, m.group(0)) \
                           for m in [re.search('thermal_zone[0-9]', d) \
                           for d in os.listdir(THERMAL_PATH)] if m]

    def _get_thermal_zone_names(self):
        '''
        Gets the thermal zone names from
        /sys/devices/virtual/thermal/thermal_zone[0-9]/type
        '''
        return [read_sys_value(os.path.join(p, 'type')) for p in self.zone_paths]

    def _get_thermal_zone_temps(self):
        '''
        Gets the thermal zone temperature values from
        /sys/devices/virtual/thermal/thermal_zone[0-9]/temp
        '''
        return([int(read_sys_value(os.path.join(p, 'temp'))) for p in self.zone_paths])

    def getTemp(self):
        return dict(zip(self._get_thermal_zone_names(), self._get_thermal_zone_temps()))


if __name__ == "__main__":
    jetson = JetsonParams()
    print(jetson.getTemp())

    # from jtop import jtop
    #
    # with jtop() as jetson:
    #     while jetson.ok():
    #         #read jetson stats
    #         print(jetson.stats)
