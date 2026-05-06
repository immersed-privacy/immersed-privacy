import glob
import sys
from sys import platform

import os

# Robustly find the simulation directory relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
simulation_path = os.path.join(current_dir, 'simulation')

if simulation_path not in sys.path:
    sys.path.append(simulation_path)

from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz