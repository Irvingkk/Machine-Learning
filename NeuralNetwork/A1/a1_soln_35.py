# uncompyle6 version 3.2.5
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.7.1 (default, Dec 14 2018, 13:28:58) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /u4/jorchard/cs489/a1/a1_solutions.py
# Compiled at: 2019-01-20 16:35:49
# Size of source mod 2**32: 7715 bytes
import numpy as np, math

class LIFNeuron(object):

    def __init__(self, tau_m=0.02, tau_ref=0.002, tau_s=0.05):
        """
        LIFNeuron(tau_m=0.02, tau_ref=0.002, tau_s=0.05)
        
        Constructor for LIFNeuron class
        
        Inputs:
         tau_m   membrane time constant, in seconds (s)
         tau_ref refractory period (s)
         tau_s   synaptic time constant (s)
        """
        self.tau_m = tau_m
        self.tau_ref = tau_ref
        self.tau_s = tau_s
        self.v = 0.0
        self.s = 0.0
        self.dvdt = 0.0
        self.dsdt = 0.0
        self.weighted_incoming_spikes = 0.0
        self.ref_remaining = 0.0
        self.v_history = []
        self.s_history = []
        self.spikes = []

    def SpikesBetween(self, t_start, t_end):
        """
        numspikes = LIFNeuron.SpikesBetween(t_start, t_end)
        
        Returns the number of times the neuron spiked between t_start and t_end.
        Specifically, it counts a spike if it occurred at t, where
        t_start <= t < t_end
        """
        sp_bool = np.logical_and(np.array(self.spikes) >= t_start, np.array(self.spikes) < t_end)
        return np.sum(sp_bool)

    def Slope(self):
        """
        LIFNeuron.Slope()
        
        Evaluates the right-hand side of the differential equations that
        govern v and s. The slopes get stored in the internal variables
          self.dvdt, and
          self.dsdt
        """
        self.dvdt = (self.s - self.v) / self.tau_m
        self.dsdt = -self.s / self.tau_s

    def Step(self, t, dt):
        """
        LIFNeuron.Step(t, dt)
        
        Updates the LIF neuron state by taking an Euler step in v and s.
        If v reaches the threshold of 1, the neuron fires an action potential
        (spike). The spike time is appended to the list self.spikes, and v
        is set to zero. After a spike, the neuron is dormant for self.tau_ref
        seconds.
        
        If the neuron is set to do spike-time interpolation (ie. self.interp==True),
        then linear interpolation is used to estimate the time that v=1.
        """
        self.s += dt * self.dsdt + self.weighted_incoming_spikes / self.tau_s
        if self.ref_remaining <= 0:
            self.v += dt * self.dvdt
        else:
            self.v = 0.0
            self.ref_remaining -= dt
        if self.v >= 1.0:
            v0 = self.v_history[-1]
            v1 = self.v
            t0 = t
            tstar = t + dt * (1.0 - v0) / (v1 - v0)
            self.spikes.append(tstar)
            self.v = 1.0
            self.ref_remaining = self.tau_ref - (self.spikes[-1] - t)
        self.v_history.append(self.v)
        self.s_history.append(self.s)
        self.weighted_incoming_spikes = 0.0

    def ReceiveSpike(self, w):
        """
        LIFNeuron.ReceiveSpike(w)
        
        Registers the arrival of a spike from a presynaptic neuron. The
        member variable self.weighted_incoming_spikes keeps track of all
        the incoming spikes, each weighted by their respective connection
        weights. It is sufficient to add them all together to tabulate the
        total incoming weighted spikes (from all presynaptic neurons).
        
        Input:
         w    is the connection weight from the presynaptic neuron.
        """
        self.weighted_incoming_spikes += w


class SpikingNetwork(object):

    def __init__(self):
        """
        SpikingNetwork()
        
        Constructor for SpikingNetwork class.
        
        The SpikingNetwork class contains a collection of neurons,
        and the connections between those neurons.
        """
        self.neurons = []
        self.connections = []
        self.t_history = []

    def AddNeuron(self, neur):
        """
        SpikingNetwork.AddNeuron(neuron)
        
        Adds a neuron to the network.
        
        Input:
         neuron is an object of type LIFNeuron or InputNeuron
        """
        self.neurons.append(neur)

    def Connect(self, pre, post, w):
        """
        SpikingNetwork.Connect(pre, post, w)
        
        Connects neuron 'pre' to neuron 'post' with a connection
        weigth of w.
        
        Each "connection" is a list of 3 numbers of the form:
         [ pre_idx, post_idx, weight ]
        where
         pre_idx is the list index of the pre-synaptic neuron,
         post_idx is the list index of the post-synaptic neuron, and
         weight is the connection weight.
        eg. self.connections = [[0,1,0.05], [1,2,0.04], [1,0,-0.2]]
        """
        self.connections.append([pre, post, w])

    def Simulate(self, T, dt=0.001):
        """
        SpikingNetwork.Simulate(T, dt=0.001)
        
        Simulates the network for T seconds by taking Euler steps
        of size dt.
        
        Inputs:
         T    how long to integrate for
         dt   time step for Euler's method
        """
        current = 0 if len(self.t_history) == 0 else self.t_history[-1]
        t_segment = np.arange(current, current + T, dt)
        for tt in t_segment:
            self.t_history.append(tt)
            for neur in self.neurons:
                neur.Slope()

            for neur in self.neurons:
                neur.Step(tt, dt)

            for pre, post, w in self.connections:
                num_spikes = self.neurons[pre].SpikesBetween(tt, tt + dt)
                self.neurons[post].ReceiveSpike(num_spikes * w)

    def AllSpikeTimes(self):
        """
        SpikingNetwork.AllSpikeTimes()
        
        Returns all the spikes of all the neurons in the network.
        Useful for making spike-raster plots of network activity.
        
        Output:
         all_spikes  a list of sublists, where each sublist holds
                     the spike times of one of the neurons
        """
        blah = []
        for neur in self.neurons:
            blah.append(np.array(neur.spikes))

        return blah
# okay decompiling a1_solutions.cpython-35.pyc
