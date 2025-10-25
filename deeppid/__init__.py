
"""
DeepPID: PID + ML playground
--------------------------------
A clean reference implementation for experimenting with classic PID control,
gain scheduling, and neural adaptive PID (GRU-based) on simple nonlinear plants.
"""
from .controllers.controllers import *
from .envs.problems import *
from .utils.sim import *
