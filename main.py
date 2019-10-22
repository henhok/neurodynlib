import sys
sys.path.append('..')
import neurodynlib as nd
from brian2.units import *
import brian2 as b2
from brian2tools import brian_plot
import matplotlib.pyplot as plt
from equation_templates import EquationHelper

# x = nd.EifNeuron()
# inputcurr = nd.input_factory.get_step_current(t_start=100, t_end=400, unit_time=ms, amplitude=20 * namp)
# x.add_tonic_current(500*pA)
# states, spikes = x.simulate_neuron(I_stim=inputcurr)
# x.plot_vm(states)

# x = EquationHelper(neuron_model='EIF', is_pyramidal=False,
#                    exc_model='AMPA_NMDA_BIEXP', inh_model='I_ALPHA')
# print(x.getMembraneEquation(return_string=True))
#
# print('-----')
#
# y = nd.neuron_factory('EIF')
# y.set_excitatory_receptors('AMPA_NMDA_BIEXP')
# y.set_inhibitory_receptors('I_ALPHA')
# print(y.get_membrane_equation(return_string=True))

x = nd.EifPyramidalCell()
x.set_excitatory_receptors('SIMPLE_E')
print(x.get_neuron_equations())