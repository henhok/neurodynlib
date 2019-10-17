import sys
sys.path.append('..')
import neurodynlib as nd
from brian2.units import *
import brian2 as b2
from brian2tools import brian_plot
import matplotlib.pyplot as plt

x = nd.AdexNeuron()
inputcurr = nd.input_factory.get_step_current(t_start=100, t_end=200, unit_time=ms, amplitude=65*pA)
statemon, spikemon = x.simulate_neuron(I_stim=inputcurr, simulation_time=300*ms)
