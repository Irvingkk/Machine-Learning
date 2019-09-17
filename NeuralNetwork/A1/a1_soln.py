# uncompyle6 version 3.2.5
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.1 (default, Dec 14 2018, 13:28:58) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/jorchard/Dropbox/teaching/cs489_neural_nets/assignments/a1/a1_solutions.py
# Compiled at: 2019-01-16 01:02:59
# Size of source mod 2**32: 7715 bytes
import numpy as np, math


def PlotSpikeRaster(st, y_range=[0, 1.]):
    '''
    PlotSpikeRaster(spiketimes, y_range=[0, 1.])

    Plots a spike raster plot for a list of arrays of spike times.

    Input:
      spiketimes is a list of arrays of spike times, like that returned
          by the function Stim2Spikes.
      y_range is a 2-tuple that holds the y-values that the raster ticks
          should be drawn between
    '''
    N = len(st)  # number of neurons

    # levels = np.linspace(y_range[0], y_range[1], N+1, endpoint=True)
    levels = np.linspace(y_range[1], y_range[0], N + 1, endpoint=True)
    for n in range(N):
        nspikes = len(st[n])
        # y = [ [levels[n]]*nspikes , [levels[n+1]]*nspikes ]
        y = [[levels[n + 1]] * nspikes, [levels[n]] * nspikes]
        # y = y_range[0] + [levels[n]]*nspikes
        plt.plot(np.vstack((st[n], st[n])), y, color=np.random.rand(3))
    plt.ylim(y_range)
    plt.xlabel('Time (s)')
    return


def GenerateSpikeTrain(rates, T, jitter=0.):
    '''
    spike_times = GenerateSpikeTrain(rates, T)

    Creates a spike train (as an array of time stamps).

    Input:
    rates is an array or list of firing rates (in Hz), one
        firing rate for each interval.
    T is an array or list (the same size as 'rates') that gives
        the time to end each interval
    jitter is a scalar that determines how much the spikes
        are randomly moved

    Output:
    spike_times is an array of times when spikes occurred

    Example: To create a spike train of 10Hz for 0.5s, followed
             by 25Hz that starts at 0.5s and ends at 2s, use

               GenerateSpikeTrain([10, 25], [0.5, 2])
    '''
    s = []
    t = 0.
    for idx in range(0, len(rates)):
        Trange = T[idx] - t
        if rates[idx] != 0:
            delta = 1. / rates[idx]
            N = rates[idx] * Trange
            times = np.arange(t + delta / 2., T[idx], delta)
            times += np.random.normal(scale=delta * jitter, size=np.shape(times))
            s.extend(times)
        t = T[idx]
    s.sort()
    return np.array(s)


class InputNeuron(object):

    def __init__(self, spiketrain):
        '''
        InputNeuron(spiketrain)

        Constructor for InputNeuron class.

        InputNeuron is a class of neuron that can be used to inject spikes into
        the network. When involved in a simulation, an InputNeuron will generate
        spikes at the times specified during its construction.

        Inputs:
         spiketrain is an array or list of spike times
        '''
        self.spikes = np.array(spiketrain)

    def SpikesBetween(self, t_start, t_end):
        '''
        numspikes = InputNeuron.SpikesBetween(t_start, t_end)

        Returns the number of times the neuron spiked between t_start and t_end.
        Specifically, it counts a spike if it occurred at t, where
        t_start <= t < t_end
        '''
        sp_bool = np.logical_and(np.array(self.spikes) >= t_start, np.array(self.spikes) < t_end)
        return np.sum(sp_bool)

    def Slope(self):
        return

    def Step(self, t, dt):
        return

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
# okay decompiling a1_solutions.pyc
