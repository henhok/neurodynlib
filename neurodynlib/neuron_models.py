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

b2.defaultclock.dt = 0.01 * b2.ms


class AdEx(object):
    """
    Implementation of the Adaptive Exponential Integrate-and-Fire model.
    See Neuronal Dynamics
    `Chapter 6 Section 1 <http://neuronaldynamics.epfl.ch/online/Ch6.S1.html>`_
    """

    # default values. (see Table 6.1, Initial Burst)
    # http://neuronaldynamics.epfl.ch/online/Ch6.S2.html#Ch6.F3
    MEMBRANE_TIME_SCALE_tau_m = 5 * b2.ms
    MEMBRANE_RESISTANCE_R = 500 * b2.Mohm
    V_REST = -70.0 * b2.mV
    V_RESET = -51.0 * b2.mV
    RHEOBASE_THRESHOLD_v_rh = -50.0 * b2.mV
    SHARPNESS_delta_T = 2.0 * b2.mV
    ADAPTATION_VOLTAGE_COUPLING_a = 0.5 * b2.nS
    ADAPTATION_TIME_CONSTANT_tau_w = 100.0 * b2.ms
    SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b = 7.0 * b2.pA

    # a technical threshold to tell the algorithm when to reset vm to v_reset
    FIRING_THRESHOLD_v_spike = -30. * b2.mV


    # This function implement Adaptive Exponential Leaky Integrate-And-Fire neuron model
    def simulate_AdEx_neuron(self,
            tau_m=MEMBRANE_TIME_SCALE_tau_m,
            R=MEMBRANE_RESISTANCE_R,
            v_rest=V_REST,
            v_reset=V_RESET,
            v_rheobase=RHEOBASE_THRESHOLD_v_rh,
            a=ADAPTATION_VOLTAGE_COUPLING_a,
            b=SPIKE_TRIGGERED_ADAPTATION_INCREMENT_b,
            v_spike=FIRING_THRESHOLD_v_spike,
            delta_T=SHARPNESS_delta_T,
            tau_w=ADAPTATION_TIME_CONSTANT_tau_w,
            I_stim=input_factory.get_zero_current(),
            simulation_time=200 * b2.ms):
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

        v_spike_str = "v>{:f}*mvolt".format(v_spike / b2.mvolt)

        # EXP-IF
        eqs = """
            dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i) - R * w)/(tau_m) : volt
            dw/dt=(a*(v-v_rest)-w)/tau_w : amp
            """

        neuron = b2.NeuronGroup(1, model=eqs, threshold=v_spike_str, reset="v=v_reset;w+=b", method="euler")

        # initial values of v and w is set here:
        neuron.v = v_rest
        neuron.w = 0.0 * b2.pA

        # Monitoring membrane voltage (v) and w
        state_monitor = b2.StateMonitor(neuron, ["v", "w"], record=True)
        spike_monitor = b2.SpikeMonitor(neuron)

        # running simulation
        b2.run(simulation_time)
        return state_monitor, spike_monitor


    def plot_adex_state(self, adex_state_monitor):
        """
        Visualizes the state variables: w-t, v-t and phase-plane w-v
        Args:
            adex_state_monitor (StateMonitor): States of "v" and "w"
        """
        plt.subplot(2, 2, 1)
        plt.plot(adex_state_monitor.t / b2.ms, adex_state_monitor.v[0] / b2.mV, lw=2)
        plt.xlabel("t [ms]")
        plt.ylabel("u [mV]")
        plt.title("Membrane potential")
        plt.subplot(2, 2, 2)
        plt.plot(adex_state_monitor.v[0] / b2.mV, adex_state_monitor.w[0] / b2.pA, lw=2)
        plt.xlabel("u [mV]")
        plt.ylabel("w [pAmp]")
        plt.title("Phase plane representation")
        plt.subplot(2, 2, 3)
        plt.plot(adex_state_monitor.t / b2.ms, adex_state_monitor.w[0] / b2.pA, lw=2)
        plt.xlabel("t [ms]")
        plt.ylabel("w [pAmp]")
        plt.title("Adaptation current")
        plt.show()


    def getting_started(self):
        """
        Simple example to get started
        """

        from neurodynlib.tools import plot_tools
        current = input_factory.get_step_current(10, 200, 1. * b2.ms, 65.0 * b2.pA)
        state_monitor, spike_monitor = self.simulate_AdEx_neuron(I_stim=current, simulation_time=300 * b2.ms)
        plot_tools.plot_voltage_and_current_traces(state_monitor, current)
        self.plot_adex_state(state_monitor)
        print("nr of spikes: {}".format(spike_monitor.count[0]))


class passive_cable(object):
    """
    Implements compartmental model of a passive cable. See Neuronal Dynamics
    `Chapter 3 Section 2 <http://neuronaldynamics.epfl.ch/online/Ch3.S2.html>`_
    """

    # DEFAULT morphological and electrical parameters
    CABLE_LENGTH = 500. * b2.um  # length of dendrite
    CABLE_DIAMETER = 2. * b2.um  # diameter of dendrite
    R_LONGITUDINAL = 0.5 * b2.kohm * b2.mm  # Intracellular medium resistance
    R_TRANSVERSAL = 1.25 * b2.Mohm * b2.mm ** 2  # cell membrane resistance (->leak current)
    E_LEAK = -70. * b2.mV  # reversal potential of the leak current (-> resting potential)
    CAPACITANCE = 0.8 * b2.uF / b2.cm ** 2  # membrane capacitance
    DEFAULT_INPUT_CURRENT = input_factory.get_step_current(2000, 3000, unit_time=b2.us, amplitude=0.2 * b2.namp)
    DEFAULT_INPUT_LOCATION = [CABLE_LENGTH / 3]  # provide an array of locations

    # print("Membrane Timescale = {}".format(R_TRANSVERSAL*CAPACITANCE))

    def simulate_passive_cable(self, current_injection_location=DEFAULT_INPUT_LOCATION, input_current=DEFAULT_INPUT_CURRENT,
                               length=CABLE_LENGTH, diameter=CABLE_DIAMETER,
                               r_longitudinal=R_LONGITUDINAL,
                               r_transversal=R_TRANSVERSAL, e_leak=E_LEAK, initial_voltage=E_LEAK,
                               capacitance=CAPACITANCE, nr_compartments=200, simulation_time=5 * b2.ms):
        """Builds a multicompartment cable and numerically approximates the cable equation.
        Args:
            t_spikes (int): list of spike times
            current_injection_location (list): List [] of input locations (Quantity, Length): [123.*b2.um]
            input_current (TimedArray): TimedArray of current amplitudes. One column per current_injection_location.
            length (Quantity): Length of the cable: 0.8*b2.mm
            diameter (Quantity): Diameter of the cable: 0.2*b2.um
            r_longitudinal (Quantity): The longitudinal (axial) resistance of the cable: 0.5*b2.kohm*b2.mm
            r_transversal (Quantity): The transversal resistance (=membrane resistance): 1.25*b2.Mohm*b2.mm**2
            e_leak (Quantity): The reversal potential of the leak current (=resting potential): -70.*b2.mV
            initial_voltage (Quantity): Value of the potential at t=0: -70.*b2.mV
            capacitance (Quantity): Membrane capacitance: 0.8*b2.uF/b2.cm**2
            nr_compartments (int): Number of compartments. Spatial discretization: 200
            simulation_time (Quantity): Time for which the dynamics are simulated: 5*b2.ms
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
            nr_compartments=100, simulation_time=2 * b2.ms)

        # provide a minimal plot
        plt.figure()
        plt.imshow(voltage_monitor.v / b2.volt)
        plt.colorbar(label="voltage")
        plt.xlabel("time index")
        plt.ylabel("location index")
        plt.title("vm at (t,x), raw data voltage_monitor.v")
        plt.show()


class exp_IF(object):
    """
    Exponential Integrate-and-Fire model.
    See Neuronal Dynamics, `Chapter 5 Section 2 <http://neuronaldynamics.epfl.ch/online/Ch5.S2.html>`_
    """

    # default values.
    MEMBRANE_TIME_SCALE_tau = 12.0 * b2.ms
    MEMBRANE_RESISTANCE_R = 20.0 * b2.Mohm
    V_REST = -65.0 * b2.mV
    V_RESET = -60.0 * b2.mV
    RHEOBASE_THRESHOLD_v_rh = -55.0 * b2.mV
    SHARPNESS_delta_T = 2.0 * b2.mV

    # a technical threshold to tell the algorithm when to reset vm to v_reset
    FIRING_THRESHOLD_v_spike = -30. * b2.mV


    def simulate_exponential_IF_neuron(self,
            tau=MEMBRANE_TIME_SCALE_tau,
            R=MEMBRANE_RESISTANCE_R,
            v_rest=V_REST,
            v_reset=V_RESET,
            v_rheobase=RHEOBASE_THRESHOLD_v_rh,
            v_spike=FIRING_THRESHOLD_v_spike,
            delta_T=SHARPNESS_delta_T,
            I_stim=input_factory.get_zero_current(),
            simulation_time=200 * b2.ms):
        """
        Implements the dynamics of the exponential Integrate-and-fire model
        Args:
            tau (Quantity): Membrane time constant
            R (Quantity): Membrane resistance
            v_rest (Quantity): Resting potential
            v_reset (Quantity): Reset value (vm after spike)
            v_rheobase (Quantity): Rheobase threshold
            v_spike (Quantity) : voltage threshold for the spike condition
            delta_T (Quantity): Sharpness of the exponential term
            I_stim (TimedArray): Input current
            simulation_time (Quantity): Duration for which the model is simulated
        Returns:
            (voltage_monitor, spike_monitor):
            A b2.StateMonitor for the variable "v" and a b2.SpikeMonitor
        """

        eqs = """
        dv/dt = (-(v-v_rest) +delta_T*exp((v-v_rheobase)/delta_T)+ R * I_stim(t,i))/(tau) : volt
        """
        neuron = b2.NeuronGroup(1, model=eqs, reset="v=v_reset", threshold="v>v_spike", method="euler")
        neuron.v = v_rest
        # monitoring membrane potential of neuron and injecting current
        voltage_monitor = b2.StateMonitor(neuron, ["v"], record=True)
        spike_monitor = b2.SpikeMonitor(neuron)

        # run the simulation
        net = b2.Network(neuron, voltage_monitor, spike_monitor)
        net.run(simulation_time)

        return voltage_monitor, spike_monitor


    def getting_started(self):
        """
        A simple example
        """

        input_current = input_factory.get_step_current(t_start=20, t_end=120, unit_time=b2.ms, amplitude=0.8 * b2.namp)
        state_monitor, spike_monitor = self.simulate_exponential_IF_neuron(
            I_stim=input_current, simulation_time=180 * b2.ms)
        plot_tools.plot_voltage_and_current_traces(
            state_monitor, input_current, title="step current", firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike)
        print("nr of spikes: {}".format(spike_monitor.count[0]))
        plt.show()


    def _min_curr_expl(self):

        durations = [1, 2, 5, 10, 20, 50, 100, 200]
        min_amp = [8.6, 4.45, 2., 1.15, .70, .48, 0.43, .4]
        i = 1
        t = durations[i]
        I_amp = min_amp[i] * b2.namp

        input_current = input_factory.get_step_current(
            t_start=10, t_end=10 + t - 1, unit_time=b2.ms, amplitude=I_amp)

        state_monitor, spike_monitor = self.simulate_exponential_IF_neuron(
            I_stim=input_current, simulation_time=(t + 20) * b2.ms)

        plot_tools.plot_voltage_and_current_traces(
            state_monitor, input_current, title="step current",
            firing_threshold=exp_IF.FIRING_THRESHOLD_v_spike, legend_location=2)
        plt.show()
        print("nr of spikes: {}".format(spike_monitor.count[0]))


class HH(object):
    """
    Implementation of a Hodging-Huxley neuron
    Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch2.S2.html
    """

    def plot_data(self, state_monitor, title=None):
        """Plots the state_monitor variables ["vm", "I_e", "m", "n", "h"] vs. time.
        Args:
            state_monitor (StateMonitor): the data to plot
            title (string, optional): plot title to display
        """

        plt.subplot(311)
        plt.plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)

        plt.xlabel("t [ms]")
        plt.ylabel("v [mV]")
        plt.grid()

        plt.subplot(312)

        plt.plot(state_monitor.t / b2.ms, state_monitor.m[0] / b2.volt, "black", lw=2)
        plt.plot(state_monitor.t / b2.ms, state_monitor.n[0] / b2.volt, "blue", lw=2)
        plt.plot(state_monitor.t / b2.ms, state_monitor.h[0] / b2.volt, "red", lw=2)
        plt.xlabel("t (ms)")
        plt.ylabel("act./inact.")
        plt.legend(("m", "n", "h"))
        plt.ylim((0, 1))
        plt.grid()

        plt.subplot(313)
        plt.plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, lw=2)
        plt.axis((
            0,
            np.max(state_monitor.t / b2.ms),
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
        El = 10.6 * b2.mV
        EK = -12 * b2.mV
        ENa = 115 * b2.mV
        gl = 0.3 * b2.msiemens
        gK = 36 * b2.msiemens
        gNa = 120 * b2.msiemens
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

    def getting_started(self):
        """
        An example to quickly get started with the Hodgkin-Huxley module.
        """
        current = input_factory.get_step_current(10, 45, b2.ms, 7.2 * b2.uA)
        state_monitor = self.simulate_HH_neuron(current, 70 * b2.ms)
        self.plot_data(state_monitor, title="HH Neuron, step current")


class LIF(object):
    """
    This file implements a leaky intergrate-and-fire (LIF) model.
    You can inject a step current or sinusoidal current into
    neuron using LIF_Step() or LIF_Sinus() methods respectively.
    Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch1.S3.html
    """

    V_REST = -70 * b2.mV
    V_RESET = -65 * b2.mV
    FIRING_THRESHOLD = -50 * b2.mV
    MEMBRANE_RESISTANCE = 10. * b2.Mohm
    MEMBRANE_TIME_SCALE = 8. * b2.ms
    ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
    __OBFUSCATION_FACTORS = [543, 622, 9307, 584, 2029, 211]

    def print_default_parameters(self):
        """
        Prints the default values
        Returns:
        """
        print("Resting potential: {}".format(LIF.V_REST))
        print("Reset voltage: {}".format(LIF.V_RESET))
        print("Firing threshold: {}".format(LIF.FIRING_THRESHOLD))
        print("Membrane resistance: {}".format(LIF.MEMBRANE_RESISTANCE))
        print("Membrane time-scale: {}".format(LIF.MEMBRANE_TIME_SCALE))
        print("Absolute refractory period: {}".format(LIF.ABSOLUTE_REFRACTORY_PERIOD))

    def simulate_LIF_neuron(self, input_current,
                            simulation_time=5 * b2.ms,
                            v_rest=V_REST,
                            v_reset=V_RESET,
                            firing_threshold=FIRING_THRESHOLD,
                            membrane_resistance=MEMBRANE_RESISTANCE,
                            membrane_time_scale=MEMBRANE_TIME_SCALE,
                            abs_refractory_period=ABSOLUTE_REFRACTORY_PERIOD):
        """Basic leaky integrate and fire neuron implementation.
        Args:
            input_current (TimedArray): TimedArray of current amplitudes. One column per current_injection_location.
            simulation_time (Quantity): Time for which the dynamics are simulated: 5ms
            v_rest (Quantity): Resting potential: -70mV
            v_reset (Quantity): Reset voltage after spike - 65mV
            firing_threshold (Quantity) Voltage threshold for spiking -50mV
            membrane_resistance (Quantity): 10Mohm
            membrane_time_scale (Quantity): 8ms
            abs_refractory_period (Quantity): 2ms
        Returns:
            StateMonitor: Brian2 StateMonitor for the membrane voltage "v"
            SpikeMonitor: Brian2 SpikeMonitor
        """

        # differential equation of Leaky Integrate-and-Fire model
        eqs = """
        dv/dt =
        ( -(v-v_rest) + membrane_resistance * input_current(t,i) ) / membrane_time_scale : volt (unless refractory)"""

        # LIF neuron using Brian2 library
        neuron = b2.NeuronGroup(
            1, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
            refractory=abs_refractory_period, method="linear")
        neuron.v = v_rest  # set initial value

        # monitoring membrane potential of neuron and injecting current
        state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
        spike_monitor = b2.SpikeMonitor(neuron)
        # run the simulation
        b2.run(simulation_time)
        return state_monitor, spike_monitor



    def _obfuscate_params(self, param_set):
        """ A helper to _obfuscate_params a parameter vector.
        Args:
            param_set:
        Returns:
            list: obfuscated list
        """
        obfuscated_factors = [LIF.__OBFUSCATION_FACTORS[i] * param_set[i] for i in range(6)]
        return obfuscated_factors

    def _deobfuscate_params(self, obfuscated_params):
        """ A helper to deobfuscate a parameter set.
        Args:
            obfuscated_params (list):
        Returns:
            list: de-obfuscated list
        """
        param_set = [obfuscated_params[i] / LIF.__OBFUSCATION_FACTORS[i] for i in range(6)]
        return param_set

    def get_random_param_set(self, random_seed=None):
        """
        creates a set of random parameters. All values are constrained to their typical range
        Returns:
            list: a list of (obfuscated) parameters. Use this vector when calling simulate_random_neuron()
        """
        random.seed(random_seed)
        v_rest = (-75. + random.randint(0, 15)) * b2.mV
        v_reset = v_rest + random.randint(-10, +10) * b2.mV
        firing_threshold = random.randint(-40, +5) * b2.mV
        membrane_resistance = random.randint(2, 15) * b2.Mohm
        membrane_time_scale = random.randint(2, 30) * b2.ms
        abs_refractory_period = random.randint(1, 7) * b2.ms
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
            simulation_time=50 * b2.ms,
            v_rest=vals[0],
            v_reset=vals[1],
            firing_threshold=vals[2],
            membrane_resistance=vals[3],
            membrane_time_scale=vals[4],
            abs_refractory_period=vals[5])
        return state_monitor, spike_monitor

    def getting_started(self):
        """
        An example to quickly get started with the LIF module.
        Returns:
        """
        # specify step current
        step_current = input_factory.get_step_current(
            t_start=100, t_end=200, unit_time=b2.ms,
            amplitude=1.2 * b2.namp)
        # run the LIF model
        (state_monitor, spike_monitor) = self.simulate_LIF_neuron(input_current=step_current, simulation_time=300 * b2.ms)

        # plot the membrane voltage
        plot_tools.plot_voltage_and_current_traces(state_monitor, step_current,
                                                   title="Step current", firing_threshold=LIF.FIRING_THRESHOLD)
        print("nr of spikes: {}".format(len(spike_monitor.t)))
        plt.show()

        # second example: sinusoidal current. note the higher resolution 0.1 * b2.ms
        sinusoidal_current = input_factory.get_sinusoidal_current(
            500, 1500, unit_time=0.1 * b2.ms,
            amplitude=2.5 * b2.namp, frequency=150 * b2.Hz, direct_current=2. * b2.namp)
        # run the LIF model
        (state_monitor, spike_monitor) = self.simulate_LIF_neuron(
            input_current=sinusoidal_current, simulation_time=200 * b2.ms)
        # plot the membrane voltage
        plot_tools.plot_voltage_and_current_traces(
            state_monitor, sinusoidal_current, title="Sinusoidal input current", firing_threshold=LIF.FIRING_THRESHOLD)
        print("nr of spikes: {}".format(spike_monitor.count[0]))
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

        t = state_monitor.t / b2.ms
        v = state_monitor.v[0] / b2.mV
        w = state_monitor.w[0] / b2.mV
        I = state_monitor.I[0] / b2.pA

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
            "g_1": 4.4 * (1 / b2.mV),
            "g_2": 8 * (1 / b2.mV),
            "g_L": 2,
            "V_1": 120 * b2.mV,
            "V_2": -84 * b2.mV,
            "V_L": -60 * b2.mV,
            "phi": 0.06666667,
            "R": 100 * b2.Gohm,
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
        self.neuron.namespace["tau"] = 15.6 * b2.ms


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
        neuron.v = v0 * b2.mV
        neuron.w = w0 * b2.mV

        # set input current
        neuron.I_e = I * b2.nA

        # record states
        rec = b2.StateMonitor(neuron, ["v", "w"], record=True)

        # run the simulation
        b2.run(tend * b2.ms)

        return (rec.t / b2.ms, rec.v[0] / b2.mV, rec.w[0] / b2.mV)

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