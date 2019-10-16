import neurodynlib as nd
from brian2.units import *

x = nd.EifNeuron()
x.getting_started()
# inputcurr = nd.input_factory.get_step_current(t_start=20, t_end=120, unit_time=ms, amplitude=0.5 * namp)
# statemon, spikemon = x.simulate_neuron(I_stim=inputcurr)