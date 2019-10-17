import sys
sys.path.append('..')
import neurodynlib as nd
from brian2.units import *
import brian2 as b2
from brian2tools import brian_plot
import matplotlib.pyplot as plt

x = nd.EifNeuron()
counts = x.plot_fi_curve()
