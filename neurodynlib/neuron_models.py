# This code is based on the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.
#
# The book is also published in print:
# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.
#
# Neuron models that were used from the exercise code repository are:
# AdEx, passive_cable, exp_IF, HH, LIF, NeuronAbstract, NeuronTypeOne, NeuronTypeTwo, fitzhugh_nagumo

import numpy as np
import matplotlib.pyplot as plt
import random
import brian2 as b2
from brian2.units import *
import neurodynlib.tools.input_factory as input_factory
from neurodynlib.tools import plot_tools, spike_tools, input_factory
from string import Template

import sys
# sys.path.append('/home/henhok/PycharmProjects/brian2modelfitting/')
# import brian2modelfitting as mofi

b2.defaultclock.dt = 0.1 * ms


class NeuronSuperclass(object):
    """
    Helper class for switching swiftly between neuron/synaptic current/synapse models in CxSystem.
    This is a preliminary version.

    Builds equation systems of the form:
    dvm/dt = ((I_NEURON_MODEL + I_SYNAPTIC + I_OTHER)/C) + I_VM_NOISE
    NEURON_MODEL_DYNAMICS
    SYNAPTIC_CURRENT_DYNAMICS
    """

    # General equation for neuron models
    membrane_eq_template = '''
    dvm/dt = (($I_NEURON_MODEL $SYN_CURRENTS $EXT_CURRENTS)/C) $VM_NOISE : volt $BRIAN2_FLAGS
    $NEURON_MODEL_EQS
    $SYN_CURRENTS_EQS
    $EXT_CURRENTS_EQS
    '''
    all_template_placeholders = ['I_NEURON_MODEL', 'SYN_CURRENTS', 'EXT_CURRENTS', 'VM_NOISE',
                                 'BRIAN2_FLAGS', 'NEURON_MODEL_EQS', 'SYN_CURRENTS_EQS',
                                 'EXT_CURRENTS_EQS']

    # Default components
    default_soma_defns = {
        # 'I_NEURON_MODEL': '',
        # 'NEURON_MODEL_EQ': '',
        # 'EXT_CURRENTS': '+ I_stim(t,i)',  # + tonic_current*(1-exp(-t/(50*msecond)))
        # 'EXT_CURRENTS_EQS': 'I_ext : amp',
        # 'VM_NOISE': '',  # + noise_sigma*xi*taum_soma**-0.5
        'BRIAN2_FLAGS': '(unless refractory)'
    }

    default_dendrite_defns = {
        # 'EXT_CURRENTS': '',
        # 'EXT_CURRENTS_EQS': '',
        # 'VM_NOISE': '',
        'BRIAN2_FLAGS': ''  # Be careful! "Unless refractory" in dendrites will not cause an error, but behavior is WRONG
    }

    default_neuron_parameters = {}

    def __init__(self, is_pyramidal=False, compartment='soma'): #, custom_strings=None):

        self.is_pyramidal = is_pyramidal
        self.compartment = compartment
        #self.custom_strings = custom_strings

        # If compartment is other than 'soma' ('a2' for example), we assume it to be a dendritic compartment
        if compartment == 'soma':
            self.compartment_type = 'soma'
        else:
            self.compartment_type = 'dendrite'

        if is_pyramidal is True:
            if self.compartment_type == 'soma':
                self.full_model_defns = dict(NeuronSuperclass.default_soma_defns)
            else:
                self.full_model_defns = dict(NeuronSuperclass.default_dendrite_defns)

            self.full_model_defns.update(self.neuron_model_defns)  # Model-specific definitions

        # Then, if we are dealing with a point neuron:
        else:
            self.full_model_defns = dict(NeuronSuperclass.default_soma_defns)
            self.full_model_defns.update(self.neuron_model_defns)  # Model-specific definitions

        # Update the generic template with neuron model-specific strings
        # _Safe_ substitute because empty keys are allowed (will be dealt with when the eqs are actually needed)
        self.neuron_eqs_template = Template(NeuronSuperclass.membrane_eq_template)
        self.neuron_eqs_template = Template(self.neuron_eqs_template.safe_substitute(self.full_model_defns))

        # Get a clean string with empty placeholders removed
        self.neuron_eqs = self.get_membrane_equation()

        # Add an empty dict for default parameters
        self.neuron_parameters = self.default_neuron_parameters

        # Add default threshold condition, reset statements and integration method
        self.threshold_condition = 'vm > V_threshold'
        self.reset_statements = 'vm = V_reset'
        self.integration_method = 'euler'

        # Add default initial values
        self.initial_values = {}

    def get_membrane_equation(self, substitute_ad_hoc=None, return_string=True):

        # Do ad hoc substitutions that don't affect the object's template
        if substitute_ad_hoc is not None:
            neuron_eqs_template2 = Template(self.neuron_eqs_template.safe_substitute(substitute_ad_hoc))
        else:
            neuron_eqs_template2 = self.neuron_eqs_template

        # Deal with extra placeholders in the eq template
        nullify_placeholders_dict = {k: '' for k in NeuronSuperclass.all_template_placeholders}
        neuron_eqs_template_wo_placeholders = neuron_eqs_template2.substitute(nullify_placeholders_dict)

        # Stringify the equations
        neuron_eqs_string = str(neuron_eqs_template_wo_placeholders)
        eq_lines = neuron_eqs_string.splitlines()
        eq_lines = [line.strip()+'\n' for line in eq_lines if len(line.strip()) > 0]
        model_membrane_equation = ''.join(eq_lines)

        if return_string is True:
            return model_membrane_equation
        else:
            #substitutables = {k: k+'_'+self.compartment for k in self.comp_specific_vars}
            #compartment_eq = b2.Equations(self.model_membrane_equation, **substitutables)
            return b2.Equations(model_membrane_equation)

    def get_neuron_equations(self):
        s = self.get_membrane_equation(return_string=True)
        return b2.Equations(s)

    def get_eqs_template(self):
        return self.neuron_eqs_template

    def get_dict(self, base_dict=None, specific_compartment='XX'):

        compartment_dict = dict(base_dict)
        substitutables = {k: k + '_' + specific_compartment for k in self.comp_specific_vars}
        compartment_dict.update(substitutables)

        return compartment_dict

    def add_tonic_current(self):
        raise NotImplementedError

    def add_vm_noise(self, noise_sigma):
        raise NotImplementedError

    def add_current(self, current_name, current_equations):
        # a.add_current('I_adaptive', '''dI_adaptive/dt = ...''')
        replace_dict = {'MISC_CURRENTS': ' + ' + current_name + ' $MISC_CURRENTS',
                        'MISC_CURRENT_EQS': current_equations + '\n$MISC_CURRENT_EQS'}
        self.neuron_eqs_template = Template(self.neuron_eqs_template.safe_substitute(replace_dict))

    def add_receptors(self, receptor_name, receptor_equations):
        # assert exc_model in NeuronSuperclass.ExcModelNames, \
        #     "Undefined excitation model!"
        # assert inh_model in NeuronSuperclass.InhModelNames, \
        #     "Undefined inhibition model!"

        # Add synaptic E/I model specific keys&strings to default strings
        # Pyramidal cells are assumed to have alpha synapses (non-zero rise time)
        # self.synaptic_excinh_model_strings = dict(NeuronSuperclass.default_synaptic_excinh_strings)
        # self.synaptic_excinh_model_strings.update(NeuronSuperclass.SynapticExcInhModels[exc_model])
        # self.synaptic_excinh_model_strings.update(NeuronSuperclass.SynapticExcInhModels[inh_model])

        # Aggregate all compartment-specific variables to a common list
        # self.comp_specific_vars = NeuronSuperclass.CompSpecificVariables[exc_model] + \
        #                           NeuronSuperclass.CompSpecificVariables[inh_model]

        raise NotImplementedError

    def get_model_definitions(self):
        # eqs, params = a.get_model_definitions()
        raise NotImplementedError

    def get_neuron_parameters(self):
        return self.neuron_parameters

    def set_neuron_parameters(self, **kwargs):
        self.neuron_parameters.update(kwargs)

    def print_default_parameters(self):
        print(self.neuron_parameters)  # Could be made fancier like in Neurodynex LIF (Resting potential: blabla, ...)

    def get_reset_statements(self):
        return self.reset_statements

    def get_threshold_condition(self):
        return self.get_threshold_condition

    def get_initial_values(self):  # Model-specific
        init_vals = dict(self.initial_values)
        vm_dict = {'vm': self.neuron_parameters['E_leak']}
        init_vals.update(vm_dict)
        return init_vals

    def simulate_neuron(self, I_stim=input_factory.get_zero_current(), simulation_time=1000*ms, **kwargs):

        neuron_parameters = dict(self.neuron_parameters)  # Make a copy of parameters; otherwise will change object params
        neuron_parameters.update(kwargs)
        refractory_period = neuron_parameters['refractory_period']

        eqs = self.get_membrane_equation(substitute_ad_hoc={'EXT_CURRENTS': '+ I_stim(t,i)'})

        # Create a neuron group
        neuron = b2.NeuronGroup(1,
                                model=eqs, namespace=neuron_parameters,
                                reset=self.reset_statements, threshold=self.threshold_condition,
                                refractory=refractory_period, method=self.integration_method)

        # Set initial values
        initial_values = self.get_initial_values()
        neuron.set_states(initial_values)

        # Set what to monitor
        state_monitor = b2.StateMonitor(neuron, ["vm"], record=True)
        spike_monitor = b2.SpikeMonitor(neuron)

        # Run the simulation
        net = b2.Network(neuron, state_monitor, spike_monitor)
        net.run(simulation_time)

        return state_monitor, spike_monitor

    def getting_started(self):
        # specify step current
        step_current = input_factory.get_step_current(t_start=100, t_end=200, unit_time=ms, amplitude=1.2 * namp)

        # run
        state_monitor, spike_monitor = self.simulate_neuron(I_stim=step_current, simulation_time=300 * ms)

        # plot the membrane voltage
        firing_threshold = self.neuron_parameters['V_threshold']
        plot_tools.plot_voltage_and_current_traces(state_monitor, step_current,
                                                   title="Step current", firing_threshold=firing_threshold)
        print("nr of spikes: {}".format(len(spike_monitor.t)))
        plt.show()

        # second example: sinusoidal current. note the higher resolution 0.1 * ms
        sinusoidal_current = input_factory.get_sinusoidal_current(
            500, 1500, unit_time=0.1 * ms,
            amplitude=2.5 * namp, frequency=150 * Hz, direct_current=2. * namp)
        # run
        state_monitor, spike_monitor = self.simulate_neuron(
            I_stim=sinusoidal_current, simulation_time=200 * ms)
        # plot the membrane voltage
        plot_tools.plot_voltage_and_current_traces(
            state_monitor, sinusoidal_current, title="Sinusoidal input current",
            firing_threshold=firing_threshold)
        print("nr of spikes: {}".format(spike_monitor.count[0]))
        plt.show()

    def plot_fi_curve(self, min_current=0*pA, max_current=1*nA):
        raise NotImplementedError


class LifNeuron(NeuronSuperclass):
    """
    This file implements a leaky intergrate-and-fire (LIF) model.
    You can inject a step current or sinusoidal current into
    neuron using LIF_Step() or LIF_Sinus() methods respectively.
    Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """

    V_REST = -70 * mV
    V_RESET = -65 * mV
    FIRING_THRESHOLD = -50 * mV
    MEMBRANE_RESISTANCE = 10. * Mohm
    MEMBRANE_TIME_SCALE = 8. * ms
    ABSOLUTE_REFRACTORY_PERIOD = 2.0 * ms
    __OBFUSCATION_FACTORS = [543, 622, 9307, 584, 2029, 211]

    default_neuron_parameters = {
            'E_leak': -70 * mV,
            'V_reset': -65 * mV,
            'V_threshold': -50 * mV,
            'g_leak': 5 * nS,
            'C': 100 * pF,
            'refractory_period': 2.0 * ms
    }

    neuron_model_defns = {'I_NEURON_MODEL': 'g_leak*(E_leak-vm)'}

    def __init__(self):
        super().__init__()

    def _obfuscate_params(self, param_set):
        """ A helper to _obfuscate_params a parameter vector.
        Args:
            param_set:
        Returns:
            list: obfuscated list
        """
        obfuscated_factors = [LifNeuron.__OBFUSCATION_FACTORS[i] * param_set[i] for i in range(6)]
        return obfuscated_factors

    def _deobfuscate_params(self, obfuscated_params):
        """ A helper to deobfuscate a parameter set.
        Args:
            obfuscated_params (list):
        Returns:
            list: de-obfuscated list
        """
        param_set = [obfuscated_params[i] / LifNeuron.__OBFUSCATION_FACTORS[i] for i in range(6)]
        return param_set

    def get_random_param_set(self, random_seed=None):
        """
        creates a set of random parameters. All values are constrained to their typical range
        Returns:
            list: a list of (obfuscated) parameters. Use this vector when calling simulate_random_neuron()
        """
        random.seed(random_seed)
        v_rest = (-75. + random.randint(0, 15)) * mV
        v_reset = v_rest + random.randint(-10, +10) * mV
        firing_threshold = random.randint(-40, +5) * mV
        membrane_resistance = random.randint(2, 15) * Mohm
        membrane_time_scale = random.randint(2, 30) * ms
        abs_refractory_period = random.randint(1, 7) * ms
        true_rand_params = [v_rest, v_reset, firing_threshold,
                            membrane_resistance, membrane_time_scale, abs_refractory_period]
        return self._obfuscate_params(true_rand_params)

    def print_obfuscated_parameters(self, obfuscated_params):
        """ Print the de-obfuscated values to the console
        Args:
            obfuscated_params:
        Returns:
        """
        true_vals = self._deobfuscate_params(obfuscated_params)
        print("Resting potential: {}".format(true_vals[0]))
        print("Reset voltage: {}".format(true_vals[1]))
        print("Firing threshold: {}".format(true_vals[2]))
        print("Membrane resistance: {}".format(true_vals[3]))
        print("Membrane time-scale: {}".format(true_vals[4]))
        print("Absolute refractory period: {}".format(true_vals[5]))

    def simulate_random_neuron(self, input_current, obfuscated_param_set):
        """
        Simulates a LIF neuron with unknown parameters (obfuscated_param_set)
        Args:
            input_current (TimedArray): The current to probe the neuron
            obfuscated_param_set (list): obfuscated parameters
        Returns:
            StateMonitor: Brian2 StateMonitor for the membrane voltage "v"
            SpikeMonitor: Biran2 SpikeMonitor
        """
        vals = self._deobfuscate_params(obfuscated_param_set)
        # run the LIF model
        state_monitor, spike_monitor = self.simulate_LIF_neuron(
            input_current,
            simulation_time=50 * ms,
            E_leak=vals[0],
            v_reset=vals[1],
            firing_threshold=vals[2],
            R=vals[3],
            tau=vals[4],
            abs_refractory_period=vals[5])
        return state_monitor, spike_monitor


class EifNeuron(NeuronSuperclass):
    """
    Exponential Integrate-and-Fire model.
    See Neuronal Dynamics, `Chapter 5 Section 2 <http://neuronaldynamics.epfl.ch/online/Ch5.S2.html>`_
    """

    # default values.
    MEMBRANE_TIME_SCALE_tau = 12.0 * ms
    MEMBRANE_RESISTANCE_R = 20.0 * Mohm
    V_REST = -65.0 * mV
    V_RESET = -60.0 * mV
    RHEOBASE_THRESHOLD_v_rh = -55.0 * mV
    SHARPNESS_delta_T = 2.0 * mV

    # a technical threshold to tell the algorithm when to reset vm to v_reset
    FIRING_THRESHOLD_v_spike = -30. * mV

    default_neuron_parameters = {
            'E_leak': -70 * mV,
            'V_reset': -65 * mV,
            'V_threshold': -50 * mV,
            'g_leak': 5 * nS,
            'C': 100 * pF,
            'DeltaT': 2 * mV,
            'refractory_period': 2.0 * ms,
            'V_cut': 20 * mV
    }

    neuron_model_defns = {'I_NEURON_MODEL': 'g_leak*(E_leak-vm) + g_leak * DeltaT * exp((vm-V_threshold) / DeltaT)'}

    def __init__(self):

        super().__init__()
        self.threshold_condition = 'vm > V_cut'

    def _min_curr_expl(self):

        durations = [1, 2, 5, 10, 20, 50, 100, 200]
        min_amp = [8.6, 4.45, 2., 1.15, .70, .48, 0.43, .4]
        i = 1
        t = durations[i]
        I_amp = min_amp[i] * b2.namp

        input_current = input_factory.get_step_current(
            t_start=10, t_end=10 + t - 1, unit_time=ms, amplitude=I_amp)

        state_monitor, spike_monitor = self.simulate_neuron(
            I_stim=input_current, simulation_time=(t + 20) * ms)

        plot_tools.plot_voltage_and_current_traces(
            state_monitor, input_current, title="step current",
            firing_threshold=EifNeuron.FIRING_THRESHOLD_v_spike, legend_location=2)
        plt.show()
        print("nr of spikes: {}".format(spike_monitor.count[0]))


class AdexNeuron(NeuronSuperclass):
    """
    Implementation of the Adaptive Exponential Integrate-and-Fire model.
    See Neuronal Dynamics
    `Chapter 6 Section 1 <http://neuronaldynamics.epfl.ch/online/Ch6.S1.html>`_
    """

    # default values. (see Table 6.1, Initial Burst)
    # http://neuronaldynamics.epfl.ch/online/Ch6.S2.html#Ch6.F3
    MEMBRANE_TIME_SCALE_tau_m = 5 * ms
    MEMBRANE_RESISTANCE_R = 500 * Mohm
    V_REST = -70.0 * mV
    V_RESET = -51.0 * mV
    RHEOBASE_THRESHOLD_v_rh = -50.0 * mV
    SHARPNESS_delta_T = 2.0 * mV
    ADAPTATION_VOLTAGE_COUPLING_a = 0.5 * b2.nS
    ADAPTATION_TIME_CONSTANT_tau_w = 100.0 * ms
    SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = 7.0 * pA

    # a technical threshold to tell the algorithm when to reset vm to v_reset
    FIRING_THRESHOLD_v_spike = -30. * mV

    default_neuron_parameters = {
            'E_leak': -70 * mV,
            'V_reset': -65 * mV,
            'V_threshold': -50 * mV,
            'g_leak': 5 * nS,
            'C': 100 * pF,
            'DeltaT': 2 * mV,
            'a': 2 * nS,
            'b': 20 * pA,
            'tau_w': 30 * ms,
            'refractory_period': 2.0 * ms,
            'V_cut': 20 * mV
    }

    neuron_model_defns = {'I_NEURON_MODEL': 'g_leak*(E_leak-vm) - w + g_leak * DeltaT * exp((vm-V_threshold) / DeltaT)',
                          'NEURON_MODEL_EQS': 'dw/dt = (a*(vm-E_leak) - w) / tau_w : amp'}

    def __init__(self):

        super().__init__()
        self.threshold_condition = 'vm > V_cut'
        self.reset_statements = 'vm = V_reset; w += b'

    # This function implement Adaptive Exponential Leaky Integrate-And-Fire neuron model
    def simulate_AdEx_neuron(self,
            tau_m=MEMBRANE_TIME_SCALE_tau_m,
            R=MEMBRANE_RESISTANCE_R,
            E_leak=V_REST,
            v_reset=V_RESET,
            VT=RHEOBASE_THRESHOLD_v_rh,
            a=ADAPTATION_VOLTAGE_COUPLING_a,
            b=SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b,
            v_spike=FIRING_THRESHOLD_v_spike,
            delta_T=SHARPNESS_delta_T,
            tau_w=ADAPTATION_TIME_CONSTANT_tau_w,
            I_stim=input_factory.get_zero_current(),
            simulation_time=200 * ms):
        r"""
        Implementation of the AdEx model with a single adaptation variable w.
        The Brian2 model equations are:
        .. math::
            \frac{dv}{dt} = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i) - R * w)/(tau_m) : volt \\
            \frac{dw}{dt} = (a*(v-v_rest)-w)/tau_w : amp

        Args:
            tau_m (Quantity): membrane time scale
            R (Quantity): membrane restistance
            v_rest (Quantity): resting potential
            v_reset (Quantity): reset potential
            v_rheobase (Quantity): rheobase threshold
            a (Quantity): Adaptation-Voltage coupling
            b (Quantity): Spike-triggered adaptation current (=increment of w after each spike)
            v_spike (Quantity): voltage threshold for the spike condition
            delta_T (Quantity): Sharpness of the exponential term
            tau_w (Quantity): Adaptation time constant
            I_stim (TimedArray): Input current
            simulation_time (Quantity): Duration for which the model is simulated
        Returns:
            (state_monitor, spike_monitor):
            A b2.StateMonitor for the variables "v" and "w" and a b2.SpikeMonitor
        """

        v_spike_str = "vm>{:f}*mvolt".format(v_spike / mvolt)

        # AdEx
        # eqs = """
        #     dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i) - R * w)/(tau_m) : volt
        #     dw/dt=(a*(v-v_rest)-w)/tau_w : amp
        #     """
        eqs = self.neuron_eqs
        C = tau_m / R
        g_leak = 1/R
        noise_sigma = 0.1*mV
        tonic_current = 0*pA
        taum_soma = tau_m

        neuron = b2.NeuronGroup(1, model=eqs, threshold=v_spike_str, reset="vm=v_reset;w+=b", method="euler")

        # initial values of v and w is set here:
        neuron.vm = E_leak
        neuron.w = 0.0 * pA

        # Monitoring membrane voltage (v) and w
        state_monitor = b2.StateMonitor(neuron, ["vm", "w"], record=True)
        spike_monitor = b2.SpikeMonitor(neuron)

        # run the simulation
        net = b2.Network(neuron, state_monitor, spike_monitor)
        net.run(simulation_time)

        return state_monitor, spike_monitor


    def plot_adex_state(self, adex_state_monitor):
        """
        Visualizes the state variables: w-t, v-t and phase-plane w-v
        Args:
            adex_state_monitor (StateMonitor): States of "v" and "w"
        """
        plt.subplot(2, 2, 1)
        plt.plot(adex_state_monitor.t / ms, adex_state_monitor.vm[0] / mV, lw=2)
        plt.xlabel("t [ms]")
        plt.ylabel("u [mV]")
        plt.title("Membrane potential")
        plt.subplot(2, 2, 2)
        plt.plot(adex_state_monitor.vm[0] / mV, adex_state_monitor.w[0] / pA, lw=2)
        plt.xlabel("u [mV]")
        plt.ylabel("w [pAmp]")
        plt.title("Phase plane representation")
        plt.subplot(2, 2, 3)
        plt.plot(adex_state_monitor.t / ms, adex_state_monitor.w[0] / pA, lw=2)
        plt.xlabel("t [ms]")
        plt.ylabel("w [pAmp]")
        plt.title("Adaptation current")
        plt.show()


    def getting_started_adex(self):
        """
        Simple example to get started
        """

        from neurodynlib.tools import plot_tools
        current = input_factory.get_step_current(10, 200, 1. * ms, 65.0 * pA)
        state_monitor, spike_monitor = self.simulate_neuron(I_stim=current, simulation_time=300 * ms)
        plot_tools.plot_voltage_and_current_traces(state_monitor, current)
        # self.plot_adex_state(state_monitor)
        print("nr of spikes: {}".format(spike_monitor.count[0]))


class HodgkinHuxleyNeuron(NeuronSuperclass):
    """
    Implementation of a Hodgkin-Huxley neuron
    Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch2.S2.html
    """

    default_neuron_parameters = {
            'E_leak': -10.6 * mV,
            'g_leak': 0.3 * msiemens,
            'C': 1 * ufarad,
            'EK': -12 * mV,
            'ENa': 115 * mV,
            'gK': 36 * msiemens,
            'gNa': 120 * msiemens,
            'refractory_period': 2.0 * ms,
            'V_spike': 20 * mV
    }

    neuron_model_defns = {
        'I_NEURON_MODEL': 'g_leak*(E_leak-vm) + gNa*m**3*h*(ENa-vm) + gK*n**4*(EK-vm)',
        'NEURON_MODEL_EQS':
        '''
        alphah = .07*exp(-.05*vm/mV)/ms : Hz
        alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
        alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
        betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
        betam = 4*exp(-.0556*vm/mV)/ms : Hz
        betan = .125*exp(-.0125*vm/mV)/ms : Hz
        dh/dt = alphah*(1-h)-betah*h : 1
        dm/dt = alpham*(1-m)-betam*m : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        '''
    }

    def __init__(self):

        super().__init__()
        self.threshold_condition = 'vm > V_spike'
        self.reset_statements = ''
        self.integration_method = 'exponential_euler'
        self.initial_values = {'m': 0.05, 'h': 0.60, 'n': 0.32}

    def plot_data(self, state_monitor, title=None):
        """Plots the state_monitor variables ["vm", "I_e", "m", "n", "h"] vs. time.
        Args:
            state_monitor (StateMonitor): the data to plot
            title (string, optional): plot title to display
        """

        plt.subplot(311)
        plt.plot(state_monitor.t / ms, state_monitor.vm[0] / mV, lw=2)

        plt.xlabel("t [ms]")
        plt.ylabel("v [mV]")
        plt.grid()

        plt.subplot(312)

        plt.plot(state_monitor.t / ms, state_monitor.m[0] / b2.volt, "black", lw=2)
        plt.plot(state_monitor.t / ms, state_monitor.n[0] / b2.volt, "blue", lw=2)
        plt.plot(state_monitor.t / ms, state_monitor.h[0] / b2.volt, "red", lw=2)
        plt.xlabel("t (ms)")
        plt.ylabel("act./inact.")
        plt.legend(("m", "n", "h"))
        plt.ylim((0, 1))
        plt.grid()

        plt.subplot(313)
        plt.plot(state_monitor.t / ms, state_monitor.I_e[0] / b2.uamp, lw=2)
        plt.axis((
            0,
            np.max(state_monitor.t / ms),
            min(state_monitor.I_e[0] / b2.uamp) * 1.1,
            max(state_monitor.I_e[0] / b2.uamp) * 1.1
        ))

        plt.xlabel("t [ms]")
        plt.ylabel("I [micro A]")
        plt.grid()

        if title is not None:
            plt.suptitle(title)

        plt.show()

    def simulate_HH_neuron(self, input_current, simulation_time):
        """A Hodgkin-Huxley neuron implemented in Brian2.
        Args:
            input_current (TimedArray): Input current injected into the HH neuron
            simulation_time (float): Simulation time [seconds]
        Returns:
            StateMonitor: Brian2 StateMonitor with recorded fields
            ["vm", "I_e", "m", "n", "h"]
        """

        # neuron parameters
        El = 10.6 * mV
        EK = -12 * mV
        ENa = 115 * mV
        gl = 0.3 * msiemens
        gK = 36 * msiemens
        gNa = 120 * msiemens
        C = 1 * b2.ufarad

        # forming HH model with differential equations
        eqs = """
        I_e = input_current(t,i) : amp
        membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + \
            gl*(El-vm) + gK*n**4*(EK-vm) : amp
        alphah = .07*exp(-.05*vm/mV)/ms    : Hz
        alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
        alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
        betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
        betam = 4*exp(-.0556*vm/mV)/ms : Hz
        betan = .125*exp(-.0125*vm/mV)/ms : Hz
        dh/dt = alphah*(1-h)-betah*h : 1
        dm/dt = alpham*(1-m)-betam*m : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        dvm/dt = membrane_Im/C : volt
        """

        neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

        # parameter initialization
        neuron.vm = 0
        neuron.m = 0.05
        neuron.h = 0.60
        neuron.n = 0.32

        # tracking parameters
        st_mon = b2.StateMonitor(neuron, ["vm", "I_e", "m", "n", "h"], record=True)

        # running the simulation
        hh_net = b2.Network(neuron)
        hh_net.add(st_mon)
        hh_net.run(simulation_time)

        return st_mon

    def getting_started_hh(self):
        """
        An example to quickly get started with the Hodgkin-Huxley module.
        """
        current = input_factory.get_step_current(10, 45, ms, 7.2 * b2.uA)
        state_monitor = self.simulate_HH_neuron(current, 70 * ms)
        self.plot_data(state_monitor, title="HH Neuron, step current")


class FitzhughNagumo(object):
    """
    This file implements functions to simulate and analyze
    Fitzhugh-Nagumo type differential equations with Brian2.
    Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch4.html
    - http://neuronaldynamics.epfl.ch/online/Ch4.S3.html.
    """

    def get_trajectory(self, v0=0., w0=0., I=0., eps=0.1, a=2.0, tend=500.):
        """Solves the following system of FitzHugh Nagumo equations
        for given initial conditions:
        dv/dt = 1/1ms * v * (1-v**2) - w + I
        dw/dt = eps * (v + 0.5 * (a - w))
        Args:
            v0: Intial condition for v [mV]
            w0: Intial condition for w [mV]
            I: Constant input [mV]
            eps: Inverse time constant of the recovery variable w [1/ms]
            a: Offset of the w-nullcline [mV]
            tend: Simulation time [ms]
        Returns:
            tuple: (t, v, w) tuple for solutions
        """

        eqs = """
        I_e : amp
        dv/dt = 1/ms * ( v * (1 - (v**2) / (mV**2) ) - w + I_e * Mohm ) : volt
        dw/dt = eps/ms * (v + 0.5 * (a * mV - w)) : volt
        """

        neuron = b2.NeuronGroup(1, eqs, method="euler")

        # state initialization
        neuron.v = v0 * mV
        neuron.w = w0 * mV

        # set input current
        neuron.I_e = I * b2.nA

        # record states
        rec = b2.StateMonitor(neuron, ["v", "w"], record=True)

        # run the simulation
        b2.run(tend * ms)

        return (rec.t / ms, rec.v[0] / mV, rec.w[0] / mV)

    def plot_flow(self, I=0., eps=0.1, a=2.0):
        """Plots the phase plane of the Fitzhugh-Nagumo model
        for given model parameters.
        Args:
            I: Constant input [mV]
            eps: Inverse time constant of the recovery variable w [1/ms]
            a: Offset of the w-nullcline [mV]
        """

        # define the interval spanned by voltage v and recovery variable w
        # to produce the phase plane
        vv = np.arange(-2.5, 2.5, 0.2)
        ww = np.arange(-2.5, 5.5, 0.2)
        (VV, WW) = np.meshgrid(vv, ww)

        # Compute derivative of v and w according to FHN equations
        # and velocity as vector norm
        dV = VV * (1. - (VV ** 2)) - WW + I
        dW = eps * (VV + 0.5 * (a - WW))
        vel = np.sqrt(dV ** 2 + dW ** 2)

        # Use quiver function to plot the phase plane
        plt.quiver(VV, WW, dV, dW, vel)

    def get_fixed_point(self, I=0., eps=0.1, a=2.0):
        """Computes the fixed point of the FitzHugh Nagumo model
        as a function of the input current I.
        We solve the 3rd order poylnomial equation:
        v**3 + V + a - I0 = 0
        Args:
            I: Constant input [mV]
            eps: Inverse time constant of the recovery variable w [1/ms]
            a: Offset of the w-nullcline [mV]
        Returns:
            tuple: (v_fp, w_fp) fixed point of the equations
        """

        # Use poly1d function from numpy to compute the
        # roots of 3rd order polynomial
        P = np.poly1d([1, 0, 1, (a - I)], variable="x")

        # take only the real root
        v_fp = np.real(P.r[np.isreal(P.r)])[0]
        w_fp = 2. * v_fp + a

        return (v_fp, w_fp)




class passive_cable(object):
    """
    Implements compartmental model of a passive cable. See Neuronal Dynamics
    `Chapter 3 Section 2 <http://neuronaldynamics.epfl.ch/online/Ch3.S2.html>`_
    """

    # DEFAULT morphological and electrical parameters
    CABLE_LENGTH = 500. * b2.um  # length of dendrite
    CABLE_DIAMETER = 2. * b2.um  # diameter of dendrite
    R_LONGITUDINAL = 0.5 * b2.kohm * b2.mm  # Intracellular medium resistance
    R_TRANSVERSAL = 1.25 * Mohm * b2.mm ** 2  # cell membrane resistance (->leak current)
    E_LEAK = -70. * mV  # reversal potential of the leak current (-> resting potential)
    CAPACITANCE = 0.8 * b2.uF / b2.cm ** 2  # membrane capacitance
    DEFAULT_INPUT_CURRENT = input_factory.get_step_current(2000, 3000, unit_time=b2.us, amplitude=0.2 * b2.namp)
    DEFAULT_INPUT_LOCATION = [CABLE_LENGTH / 3]  # provide an array of locations

    # print("Membrane Timescale = {}".format(R_TRANSVERSAL*CAPACITANCE))

    def simulate_passive_cable(self, current_injection_location=DEFAULT_INPUT_LOCATION, input_current=DEFAULT_INPUT_CURRENT,
                               length=CABLE_LENGTH, diameter=CABLE_DIAMETER,
                               r_longitudinal=R_LONGITUDINAL,
                               r_transversal=R_TRANSVERSAL, e_leak=E_LEAK, initial_voltage=E_LEAK,
                               capacitance=CAPACITANCE, nr_compartments=200, simulation_time=5 * ms):
        """Builds a multicompartment cable and numerically approximates the cable equation.
        Args:
            t_spikes (int): list of spike times
            current_injection_location (list): List [] of input locations (Quantity, Length): [123.*b2.um]
            input_current (TimedArray): TimedArray of current amplitudes. One column per current_injection_location.
            length (Quantity): Length of the cable: 0.8*b2.mm
            diameter (Quantity): Diameter of the cable: 0.2*b2.um
            r_longitudinal (Quantity): The longitudinal (axial) resistance of the cable: 0.5*b2.kohm*b2.mm
            r_transversal (Quantity): The transversal resistance (=membrane resistance): 1.25*Mohm*b2.mm**2
            e_leak (Quantity): The reversal potential of the leak current (=resting potential): -70.*mV
            initial_voltage (Quantity): Value of the potential at t=0: -70.*mV
            capacitance (Quantity): Membrane capacitance: 0.8*b2.uF/b2.cm**2
            nr_compartments (int): Number of compartments. Spatial discretization: 200
            simulation_time (Quantity): Time for which the dynamics are simulated: 5*ms
        Returns:
            (StateMonitor, SpatialNeuron): The state monitor contains the membrane voltage in a
            Time x Location matrix. The SpatialNeuron object specifies the simulated neuron model
            and gives access to the morphology. You may want to use those objects for
            spatial indexing: myVoltageStateMonitor[mySpatialNeuron.morphology[0.123*b2.um]].v
        """
        assert isinstance(input_current, b2.TimedArray), "input_current is not of type TimedArray"
        assert input_current.values.shape[1] == len(current_injection_location), \
            "number of injection_locations does not match nr of input currents"

        cable_morphology = b2.Cylinder(diameter=diameter, length=length, n=nr_compartments)
        # Im is transmembrane current
        # Iext is  injected current at a specific position on dendrite
        EL = e_leak
        RT = r_transversal
        eqs = """
        Iext = current(t, location_index): amp (point current)
        location_index : integer (constant)
        Im = (EL-v)/RT : amp/meter**2
        """
        cable_model = b2.SpatialNeuron(morphology=cable_morphology, model=eqs, Cm=capacitance, Ri=r_longitudinal)
        monitor_v = b2.StateMonitor(cable_model, "v", record=True)

        # inject all input currents at the specified location:
        nr_input_locations = len(current_injection_location)
        input_current_0 = np.insert(input_current.values, 0, 0., axis=1) * b2.amp  # insert default current: 0. [amp]
        current = b2.TimedArray(input_current_0, dt=input_current.dt * b2.second)
        for current_index in range(nr_input_locations):
            insert_location = current_injection_location[current_index]
            compartment_index = int(np.floor(insert_location / (length / nr_compartments)))
            # next line: current_index+1 because 0 is the default current 0Amp
            cable_model.location_index[compartment_index] = current_index + 1

        # set initial values and run for 1 ms
        cable_model.v = initial_voltage
        b2.run(simulation_time)
        return monitor_v, cable_model

    def getting_started(self):
        """A simple code example to get started.
        """
        current = input_factory.get_step_current(500, 510, unit_time=b2.us, amplitude=3. * b2.namp)
        voltage_monitor, cable_model = self.simulate_passive_cable(
            length=0.5 * b2.mm, current_injection_location=[0.1 * b2.mm], input_current=current,
            nr_compartments=100, simulation_time=2 * ms)

        # provide a minimal plot
        plt.figure()
        plt.imshow(voltage_monitor.v / b2.volt)
        plt.colorbar(label="voltage")
        plt.xlabel("time index")
        plt.ylabel("location index")
        plt.title("vm at (t,x), raw data voltage_monitor.v")
        plt.show()


# TODO? Part of neuron_type/neurons.py missing
class NeuronAbstract(object):
    """
    This file implements a type I and a type II model from
    the abstract base class NeuronAbstract.
    You can inject step currents and plot the responses,
    as well as get firing rates.
    Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch4.S4.html

    Abstract base class for both neuron types.
    This stores its own recorder and network, allowing
    each neuron to be run several times with changing
    currents while keeping the same neurogroup object
    and network internally.
    """

    def __init__(self):
        self._make_neuron()
        self.rec = b2.StateMonitor(self.neuron, ["v", "w", "I"], record=True)
        self.net = b2.Network([self.neuron, self.rec])
        self.net.store()

    def _make_neuron(self):
        """Abstract function, which creates neuron attribute for this class."""

        raise NotImplementedError

    def get_neuron_type(self):
        """
        Type I or II.
        Returns:
            type as a string "Type I" or "Type II"
        """
        return self._get_neuron_type()

    def _get_neuron_type(self):
        """Just a trick to have the underlying function NOT being documented by sphinx
        (because this function's name starts with _)"""
        raise NotImplementedError

    def run(self, input_current, simtime):
        """Runs the neuron for a given current.
        Args:
            input_current (TimedArray): Input current injected into the neuron
            simtime (Quantity): Simulation time in correct Brian units.
        Returns:
            StateMonitor: Brian2 StateMonitor with input current (I) and
            voltage (V) recorded
        """

        self.net.restore()
        self.neuron.namespace["input_current"] = input_current

        # run the simulation
        self.net.run(simtime)

        return self.rec

    def plot_data(self, state_monitor, title=None, show=True):
        """Plots a TimedArray for values I, v and w
        Args:
            state_monitor (StateMonitor): the data to plot. expects ["v", "w", "I"] and (by default) "t"
            title (string, optional): plot title to display
            show (bool, optional): call plt.show for the plot
        Returns:
            StateMonitor: Brian2 StateMonitor with input current (I) and
                voltage (V) recorded
        """

        t = state_monitor.t / ms
        v = state_monitor.v[0] / mV
        w = state_monitor.w[0] / mV
        I = state_monitor.I[0] / pA

        # plot voltage time series
        plt.figure()
        plt.subplot(311)
        plt.plot(t, v, lw=2)
        plt.xlabel("t [ms]")
        plt.ylabel("v [mV]")
        plt.grid()

        # plot activation and inactivation variables
        plt.subplot(312)
        plt.plot(t, w, "k", lw=2)
        plt.xlabel("t [ms]")
        plt.ylabel("w [mV]")
        plt.grid()

        # plot current
        plt.subplot(313)
        plt.plot(t, I, lw=2)
        plt.axis((0, t.max(), 0, I.max() * 1.1))
        plt.xlabel("t [ms]")
        plt.ylabel("I [pA]")
        plt.grid()

        if title is not None:
            plt.suptitle(title)

        if show:
            plt.show()


class _NeuronTypeOne(NeuronAbstract):

    def _get_neuron_type(self):
        return "Type I"

    def _make_neuron(self):
        """Sets the self.neuron attribute."""

        # neuron parameters
        pars = {
            "g_1": 4.4 * (1 / mV),
            "g_2": 8 * (1 / mV),
            "g_L": 2,
            "V_1": 120 * mV,
            "V_2": -84 * mV,
            "V_L": -60 * mV,
            "phi": 0.06666667,
            "R": 100 * Gohm,
        }

        # forming the neuron model using differential equations
        eqs = """
        I = input_current(t,i) : amp
        winf = (0.5*mV)*( 1 + tanh((v-12*mV)/(17*mV)) ) : volt
        tau = (1*ms)/cosh((v-12*mV)/(2*17*mV)) : second
        m = (0.5*mV)*(1+tanh((v+1.2*mV)/(18*mV))) : volt
        dv/dt = (-g_1*m*(v-V_1) - g_2*w*(v-V_2) - g_L*(v-V_L) \
            + I*R)/(20*ms) : volt
        dw/dt = phi*(winf-w)/tau : volt
        """

        self.neuron = b2.NeuronGroup(1, eqs, method="euler")
        self.neuron.v = pars["V_L"]
        self.neuron.namespace.update(pars)


class _NeuronTypeTwo(NeuronAbstract):

    def _get_neuron_type(self):
        return "Type II"

    def _make_neuron(self):
        """Sets the self.neuron attribute."""

        # forming the neuron model using differential equations
        eqs = """
        I = input_current(t,i) : amp
        dv/dt = (v - (v**3)/(3*mvolt*mvolt) - w + I*Gohm)/ms : volt
        dw/dt = (a*(v+0.7*mvolt)-w)/tau : volt
        """

        self.neuron = b2.NeuronGroup(1, eqs, method="euler")
        self.neuron.v = 0

        self.neuron.namespace["a"] = 1.25
        self.neuron.namespace["tau"] = 15.6 * ms


# TODO:
# - Make neuron parameters more accessible
# - run/simulate_XX_neuron - use consistent naming
# - run: ability to choose which variables to monitor
# - Joustavampi templating systeemi
# - Default neuroniparametrit pitää pystyä vaihtamaan ja sitten mahdollisuus overraidata simulate(param1, param2,...)

if __name__ == '__main__':
    # fixed_values = {'C': 110 * pF, 'g_leak': 3.1 * nS, 'E_leak': -70 * mV, 'Vcut': 20 * mV, 'refr_time': 4 * ms}
    # a = NeuronSuperclass({'I_NEURON_MODEL': 'kissa', 'NEURON_MODEL_EQ': 'dkissa/dt = mau'})
    # print(a.get_membrane_equation(return_string=True))
    from brian2tools import *

    # a = exp_IF()
    # # a.getting_started()
    # inputcurr = input_factory.get_step_current(t_start=20, t_end=120, unit_time=ms, amplitude=0.5 * namp)
    # states, spikes = a.simulate_exponential_IF_neuron(I_stim=inputcurr)
    # brian_plot(states)
    # plt.ylim([-70, 20])
    # plt.show()

    # a = AdEx()
    # inputcurr = input_factory.get_step_current(t_start=20, t_end=120, unit_time=ms, amplitude=0.2 * namp)
    # states, spikes = a.simulate_AdEx_neuron(I_stim=inputcurr)
    # a.getting_started()
    # a.plot_adex_state(states)
    # plot_state(states.t, states.vm[0])
    # plt.ylim([-80, 20])
    # plt.show()

    a = LifNeuron()
    a.simulate_LIF_neuron()
    a.getting_started()