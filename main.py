import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
import pickle
import os
from datetime import datetime

bp.math.set_platform('cpu')

def save_network_weights(net, filename=None):
    """Save network weights and parameters to a file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stdp_model_{timestamp}.pkl"
    
    # Extract all relevant weights and parameters
    model_data = {
        'network_params': {
            'n_input': net.n_input,
            'n_hidden': net.n_hidden,
            'n_output': net.n_output,
        },
        'weights': {},
        'biases': {},
        'neuron_params': {}
    }
    
    # Save synaptic weights
    if hasattr(net.syn_ih, 'w'):
        model_data['weights']['input_hidden'] = net.syn_ih.w.value.copy()
    elif hasattr(net.syn_ih, 'weights'):
        model_data['weights']['input_hidden'] = net.syn_ih.weights.value.copy()
    
    if hasattr(net.syn_ho, 'weights'):
        model_data['weights']['hidden_output'] = net.syn_ho.weights.value.copy()
    
    if hasattr(net.syn_hh, 'weights'):
        model_data['weights']['hidden_hidden'] = net.syn_hh.weights.value.copy()
    
    if hasattr(net.syn_oo, 'weights'):
        model_data['weights']['output_output'] = net.syn_oo.weights.value.copy()
    
    # Save neuron parameters (heterogeneity)
    model_data['neuron_params']['hidden'] = {
        'V_rest': net.hidden_layer.V_rest.copy(),
        'V_th_base': net.hidden_layer.V_th_base.copy(),
        'V_reset': net.hidden_layer.V_reset.copy(),
        'tau': net.hidden_layer.tau.copy(),
        'tau_ref': net.hidden_layer.tau_ref.copy(),
        'tau_adapt': net.hidden_layer.tau_adapt.copy(),
        'a_adapt': net.hidden_layer.a_adapt.copy(),
        'excitability': net.hidden_layer.excitability.value.copy()
    }
    
    model_data['neuron_params']['output'] = {
        'V_rest': net.output_layer.V_rest.copy(),
        'V_th_base': net.output_layer.V_th_base.copy(),
        'V_reset': net.output_layer.V_reset.copy(),
        'tau': net.output_layer.tau.copy(),
        'tau_ref': net.output_layer.tau_ref.copy(),
        'tau_adapt': net.output_layer.tau_adapt.copy(),
        'a_adapt': net.output_layer.a_adapt.copy(),
        'excitability': net.output_layer.excitability.value.copy(),
        'bias': net.output_layer.bias.value.copy()
    }
    
    # Save connection matrices
    model_data['connections'] = {
        'input_hidden': net.syn_ih.conn_mat.copy() if hasattr(net.syn_ih, 'conn_mat') else None,
        'hidden_output': net.syn_ho.conn_mat.copy() if hasattr(net.syn_ho, 'conn_mat') else None,
    }
    
    # Save to file
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {filename}")
    return filename


def load_network_weights(filename, use_stdp=True):
    """Load network weights and parameters from a file"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    # Create new network with same architecture
    net = SpikingDigitClassifier(
        n_input=model_data['network_params']['n_input'],
        n_hidden=model_data['network_params']['n_hidden'],
        use_stdp=use_stdp
    )
    
    # Restore synaptic weights
    if 'input_hidden' in model_data['weights']:
        if hasattr(net.syn_ih, 'w'):
            net.syn_ih.w.value = model_data['weights']['input_hidden'].copy()
        elif hasattr(net.syn_ih, 'weights'):
            net.syn_ih.weights.value = model_data['weights']['input_hidden'].copy()
    
    if 'hidden_output' in model_data['weights']:
        net.syn_ho.weights.value = model_data['weights']['hidden_output'].copy()
    
    if 'hidden_hidden' in model_data['weights']:
        net.syn_hh.weights.value = model_data['weights']['hidden_hidden'].copy()
    
    if 'output_output' in model_data['weights']:
        net.syn_oo.weights.value = model_data['weights']['output_output'].copy()
    
    # Restore neuron parameters
    if 'hidden' in model_data['neuron_params']:
        params = model_data['neuron_params']['hidden']
        net.hidden_layer.V_rest = params['V_rest'].copy()
        net.hidden_layer.V_th_base = params['V_th_base'].copy()
        net.hidden_layer.V_reset = params['V_reset'].copy()
        net.hidden_layer.tau = params['tau'].copy()
        net.hidden_layer.tau_ref = params['tau_ref'].copy()
        net.hidden_layer.tau_adapt = params['tau_adapt'].copy()
        net.hidden_layer.a_adapt = params['a_adapt'].copy()
        net.hidden_layer.excitability.value = params['excitability'].copy()
    
    if 'output' in model_data['neuron_params']:
        params = model_data['neuron_params']['output']
        net.output_layer.V_rest = params['V_rest'].copy()
        net.output_layer.V_th_base = params['V_th_base'].copy()
        net.output_layer.V_reset = params['V_reset'].copy()
        net.output_layer.tau = params['tau'].copy()
        net.output_layer.tau_ref = params['tau_ref'].copy()
        net.output_layer.tau_adapt = params['tau_adapt'].copy()
        net.output_layer.a_adapt = params['a_adapt'].copy()
        net.output_layer.excitability.value = params['excitability'].copy()
        net.output_layer.bias.value = params['bias'].copy()
    
    # Restore connection matrices
    if 'connections' in model_data and model_data['connections']['input_hidden'] is not None:
        net.syn_ih.conn_mat = model_data['connections']['input_hidden'].copy()
    
    if 'connections' in model_data and model_data['connections']['hidden_output'] is not None:
        net.syn_ho.conn_mat = model_data['connections']['hidden_output'].copy()
    
    print(f"Model loaded from: {filename}")
    return net


def list_saved_models():
    """List all saved model files in the current directory"""
    model_files = [f for f in os.listdir('.') if f.startswith('stdp_model_') and f.endswith('.pkl')]
    return sorted(model_files)


def prompt_user_for_model():
    """Prompt user to load existing model or train new one"""
    saved_models = list_saved_models()
    
    if not saved_models:
        print("\nNo saved models found.")
        return None
    
    print("\n" + "="*50)
    print("Saved Models Found:")
    print("="*50)
    for i, model_file in enumerate(saved_models, 1):
        # Extract timestamp and format nicely
        timestamp_str = model_file.replace('stdp_model_', '').replace('.pkl', '')
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. {model_file} (created: {formatted_time})")
        except:
            print(f"{i}. {model_file}")
    
    print(f"\n0. Train a new model")
    
    while True:
        try:
            choice = input("\nEnter your choice (0 to train new, or model number to load): ")
            choice = int(choice)
            
            if choice == 0:
                return None
            elif 1 <= choice <= len(saved_models):
                return saved_models[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Neuron Models
class LIFNeuron(bp.dyn.NeuDyn):
    """Leaky Integrate-and-Fire neuron with adaptive threshold and heterogeneity"""
    def __init__(self, size, V_rest=-65., V_th=-50., V_reset=-65., tau=20., 
                 tau_ref=5., tau_adapt=100., a_adapt=0.01, bias=None,
                 heterogeneity=0.1):
        super().__init__(size=size)
        
        # Store base parameters
        self.V_rest_base = V_rest
        self.V_th_base_value = V_th
        self.V_reset_base = V_reset
        self.tau_base = tau
        self.tau_ref_base = tau_ref
        self.tau_adapt_base = tau_adapt
        self.a_adapt_base = a_adapt
        self.heterogeneity = heterogeneity
        
        # Create heterogeneous parameters for each neuron
        # Voltage parameters with small variations
        self.V_rest = V_rest + bm.random.normal(0, heterogeneity * 5, size)
        self.V_th_base = V_th + bm.random.normal(0, heterogeneity * 3, size)
        self.V_reset = V_reset + bm.random.normal(0, heterogeneity * 5, size)
        
        # Time constants with multiplicative variations (to keep them positive)
        self.tau = tau * (1 + bm.random.normal(0, heterogeneity, size))
        self.tau = bm.maximum(self.tau, 1.0)  # Ensure tau stays positive
        
        self.tau_ref = tau_ref * (1 + bm.random.normal(0, heterogeneity * 0.5, size))
        self.tau_ref = bm.maximum(self.tau_ref, 0.1)  # Ensure positive
        
        self.tau_adapt = tau_adapt * (1 + bm.random.normal(0, heterogeneity, size))
        self.tau_adapt = bm.maximum(self.tau_adapt, 1.0)  # Ensure positive
        
        # Adaptation strength with variation
        self.a_adapt = a_adapt * (1 + bm.random.normal(0, heterogeneity * 2, size))
        self.a_adapt = bm.maximum(self.a_adapt, 0.0)  # Ensure non-negative
        
        # Initialize variables
        self.V = bm.Variable(self.V_rest.copy())  # Start at resting potential
        self.V_th = bm.Variable(self.V_th_base.copy())  # Dynamic threshold
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.refractory = bm.Variable(bm.zeros(size))
        self.input = bm.Variable(bm.zeros(size))
        self.adaptation = bm.Variable(bm.zeros(size))
        
        # Add bias
        if bias is None:
            self.bias = bm.Variable(bm.zeros(size))
        else:
            self.bias = bm.Variable(bias)
            
        # Optional: Add intrinsic excitability variations
        # This creates neurons that are naturally more or less excitable
        self.excitability = bm.Variable(
            1.0 + bm.random.normal(0, heterogeneity * 0.5, size)
        )
        self.excitability.value = bm.clip(self.excitability, 0.5, 1.5)
            
    def update(self, inp=None):
        """Integrate one time step (units: ms).  Works for any runner dt."""
        # Simulation time step in ms
        dt = float(getattr(self, 'dt', bm.get_dt()))  # brainpy exposes dt

        # Total synaptic + bias input (add optional external 'inp')
        total_input = (self.input.value + self.bias.value) * self.excitability.value
        if inp is not None:
            total_input += inp
        # Small noise to break symmetry
        total_input += bm.random.normal(0., 0.2, self.num)

        # Refractory counter – decrement by *dt* (time) not by one step
        self.refractory.value = bm.maximum(self.refractory - dt, 0.)

        not_refractory = self.refractory <= 0.

        # Membrane dynamics  dV/dt = (Vrest – V + I)/tau
        # Euler step: V <- V + dt * dV/dt
        dV = ((self.V_rest - self.V) + total_input) / self.tau
        V_new = bm.where(not_refractory, self.V + dt * dV, self.V)

        # Spike detection & reset
        self.spike.value = bm.logical_and(V_new >= self.V_th, not_refractory)
        V_new = bm.where(self.spike, self.V_reset, V_new)

        # Spike‑frequency adaptation
        #   τ_a dA/dt = −A    ⇒  A <- A − dt * A / τ_a
        self.adaptation.value = self.adaptation - dt * self.adaptation / self.tau_adapt
        #   Increment A when the neuron fires
        self.adaptation.value = bm.where(self.spike,
                                         self.adaptation + self.a_adapt,
                                         self.adaptation)
        # Dynamic threshold
        self.V_th.value = self.V_th_base + self.adaptation

        # Enter refractory state (time, not steps!)
        self.refractory.value = bm.where(self.spike, self.tau_ref, self.refractory)

        # Commit membrane potential and clear synaptic accumulator
        self.V.value = V_new
        self.input[:] = 0.0

        return self.spike.value

class PoissonInput(bp.dyn.NeuDyn):
    """Poisson spike generator for input encoding"""
    def __init__(self, size, freq_max=100.):
        super().__init__(size=size)
        self.freq_max = freq_max
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.rates = bm.Variable(bm.zeros(size))
        
    def update(self, rates=None):
        if rates is not None:
            self.rates.value = rates
        
        # Correctly calculate spike probability based on rate (in Hz) and time step (in ms)
        prob = self.rates * (bm.dt / 1000.)
        self.spike.value = bm.random.rand(self.num) < prob

        
        return self.spike.value

# Custom Connection Class
class MatrixConn(bp.conn.TwoEndConnector):
    """Custom connection class that uses a pre-defined connection matrix"""
    def __init__(self, conn_mat):
        self.conn_mat = conn_mat
        # Initialize the parent class without size arguments
        super().__init__()
        # Set sizes manually
        self.pre_size = conn_mat.shape[0]
        self.post_size = conn_mat.shape[1]
    
    def __call__(self, pre_size, post_size):
        # This method is called by BrainPy to build the connection
        # Handle both tuple and int inputs
        if isinstance(pre_size, tuple):
            pre_size = pre_size[0] if len(pre_size) == 1 else pre_size
        if isinstance(post_size, tuple):
            post_size = post_size[0] if len(post_size) == 1 else post_size
            
        # For comparison, also handle our stored sizes
        expected_pre = self.pre_size[0] if isinstance(self.pre_size, tuple) else self.pre_size
        expected_post = self.post_size[0] if isinstance(self.post_size, tuple) else self.post_size
        
        assert pre_size == expected_pre, f"Pre size mismatch: {pre_size} != {expected_pre}"
        assert post_size == expected_post, f"Post size mismatch: {post_size} != {expected_post}"
        
        i_indices, j_indices = np.where(self.conn_mat)
        return bp.connect.IJConn(i=i_indices, j=j_indices)(pre_size, post_size)
    
    def require(self, key):
        if key == 'conn_mat':
            return self.conn_mat
        else:
            return super().require(key)

# Synapse Models
class ExponentialSynapse(bp.dyn.SynConn):
    """Exponential synapse with synaptic delay"""
    def __init__(self, pre, post, conn, g_max=1., tau=10., delay=1.):
        super().__init__(pre=pre, post=post, conn=conn)
        
        # Parameters
        self.g_max = g_max
        self.tau = tau
        
        # Correctly calculate the delay in simulation steps
        self.delay_step = int(np.round(delay / bm.get_dt()))
        
        # Get connection matrix
        self.conn_mat = self.conn.require('conn_mat')
        
        # Initialize weights
        if g_max > 0:
            self.weights = bm.Variable(
                self.conn_mat * bm.random.uniform(0.3, 1.0, self.conn_mat.shape) * g_max
            )
        else:
            self.weights = bm.Variable(self.conn_mat * g_max)
        
        # Synaptic conductance
        self.g = bm.Variable(bm.zeros(post.num))
        
        # Spike history buffer for delays (if delay is used)
        if self.delay_step > 0:
            self.spike_buffer = bm.Variable(bm.zeros((self.delay_step, pre.num), dtype=bool))
            
    def update(self):
        # Update synaptic conductance with correct exponential decay
        self.g.value *= bm.exp(-bm.dt / self.tau)
        
        # Get spikes (either delayed or current)
        if self.delay_step > 0:
            # Get the spikes that arrived from the past
            delayed_spikes = self.spike_buffer[0]
            
            # Shift the buffer to make room for new spikes
            # Note: This explicit loop is clear and works fine with JIT for fixed loop counts.
            for i in range(self.delay_step - 1):
                self.spike_buffer[i] = self.spike_buffer[i+1]
            
            # Add the current spike to the end of the buffer for the future
            self.spike_buffer[-1] = self.pre.spike
        else:
            # No delay
            delayed_spikes = self.pre.spike
        
        # REMOVED the "if bm.any(delayed_spikes):" condition.
        # This calculation is now performed unconditionally, which is JIT-compatible.
        # If delayed_spikes is all False, the result of the dot product will be zero.
        spike_input = bm.dot(delayed_spikes.astype(bm.float32), 
                           self.weights.astype(bm.float32))
        self.g.value += spike_input
        
        # Apply conductance to post-synaptic input
        self.post.input.value += self.g.value

class STDPSynapse(bp.dyn.SynConn):
    """Spike-Timing Dependent Plasticity synapse"""
    def __init__(self, pre, post, conn, g_max=1., tau=10., 
                 tau_pre=20., tau_post=20., A_pre=0.01, A_post=0.01,
                 w_min=0., w_max=2.):
        super().__init__(pre=pre, post=post, conn=conn)
        
        self.tau = tau
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_pre = A_pre
        self.A_post = A_post
        self.w_min = w_min
        self.w_max = w_max
        
        self.conn_mat = self.conn.require('conn_mat')
        
        self.w = bm.Variable(
            self.conn_mat * bm.random.uniform(0.5 * g_max, g_max, self.conn_mat.shape)
        )
        self.g = bm.Variable(bm.zeros(post.num))
        self.trace_pre = bm.Variable(bm.zeros(pre.num))
        self.trace_post = bm.Variable(bm.zeros(post.num))
        
        
    def update(self):
        # Update synaptic conductance with correct decay
        self.g.value *= bm.exp(-bm.dt / self.tau)
        
        # Add input from pre-synaptic spikes
        spike_input = bm.dot(self.pre.spike.astype(bm.float32), 
                           (self.conn_mat * self.w).astype(bm.float32))
        self.g.value += spike_input
        
        # Update STDP traces with correct decay
        self.trace_pre.value *= bm.exp(-bm.dt / self.tau_pre)
        self.trace_post.value *= bm.exp(-bm.dt / self.tau_post)
        
        # Update traces for spiking neurons
        self.trace_pre.value = bm.where(self.pre.spike, 
                                       self.trace_pre + 1.0, 
                                       self.trace_pre)
        self.trace_post.value = bm.where(self.post.spike, 
                                        self.trace_post + 1.0, 
                                        self.trace_post)
        
        # STDP weight updates using vectorized operations
        pre_spike_expanded = self.pre.spike[:, None].astype(bm.float32)
        post_trace_expanded = self.trace_post[None, :].astype(bm.float32)
        ltd_update = -self.A_post * pre_spike_expanded * post_trace_expanded * self.w
        
        pre_trace_expanded = self.trace_pre[:, None].astype(bm.float32)
        post_spike_expanded = self.post.spike[None, :].astype(bm.float32)
        ltp_update = self.A_pre * pre_trace_expanded * post_spike_expanded * (self.w_max - self.w)
        
        weight_update = (ltd_update + ltp_update) * self.conn_mat
        self.w.value += weight_update
        
        self.w.value = bm.clip(self.w, self.w_min, self.w_max)
        
        self.post.input.value += self.g.value

# Network Model
class SpikingDigitClassifier(bp.Network):
    """Multi-layer spiking neural network for digit classification with convolution-style input layer"""
    def __init__(self, n_input=64, n_output=10, use_stdp=False):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        # === Convolution-style input to hidden setup ===
        kernel_size = 3
        stride = 1
        input_side = 8  # for 8x8 digit images
        output_side = (input_side - kernel_size) // stride + 1
        n_hidden = output_side ** 2  # 6x6 = 36
        self.n_hidden = n_hidden

        # Input layer (Poisson neurons)
        self.input_layer = PoissonInput(size=n_input, freq_max=200.)

        # Hidden layer
        self.hidden_layer = LIFNeuron(
            size=n_hidden,
            V_rest=-65., V_th=-52., V_reset=-65.,
            tau=20., tau_ref=3.,
            tau_adapt=100., a_adapt=0.005,
            heterogeneity=0.1
        )

        # Output layer
        output_bias = bm.random.uniform(-3, 3, n_output)
        self.output_layer = LIFNeuron(
            size=n_output,
            V_rest=-65., V_th=-54., V_reset=-65.,
            tau=15., tau_ref=2.,
            tau_adapt=50., a_adapt=0.01,
            bias=output_bias,
            heterogeneity=0.2
        )

        # Build conv-style connection matrix
        conn_mat = np.zeros((n_input, n_hidden), dtype=bool)
        h_idx = 0
        for y in range(0, input_side - kernel_size + 1, stride):
            for x in range(0, input_side - kernel_size + 1, stride):
                for dy in range(kernel_size):
                    for dx in range(kernel_size):
                        input_idx = (y + dy) * input_side + (x + dx)
                        conn_mat[input_idx, h_idx] = True
                h_idx += 1

        # Create custom connection using our MatrixConn class
        conn_ih = MatrixConn(conn_mat)
        
        if use_stdp:
            self.syn_ih = STDPSynapse(
                pre=self.input_layer,
                post=self.hidden_layer,
                conn=conn_ih,
                g_max=30.0,
                tau=10.0,
                A_pre=0.01,
                A_post=0.01
            )
        else:
            self.syn_ih = ExponentialSynapse(
                pre=self.input_layer,
                post=self.hidden_layer,
                conn=conn_ih,
                g_max=30.0,
                tau=10.0
            )

        # Hidden to Output: sparse connectivity
        conn_ho = bp.conn.FixedProb(prob=0.7)
        self.syn_ho = ExponentialSynapse(
            pre=self.hidden_layer,
            post=self.output_layer,
            conn=conn_ho,
            g_max=25.0,
            tau=15.0
        )

        # Class-specific weight enhancement
        if hasattr(self.syn_ho, 'weights'):
            n_hidden_per_class = n_hidden // n_output
            for i in range(n_output):
                start_idx = i * n_hidden_per_class
                end_idx = (i + 1) * n_hidden_per_class
                self.syn_ho.weights[start_idx:end_idx, i] *= 2.0
                self.syn_ho.weights[:, i] *= bm.random.uniform(0.5, 1.5, self.syn_ho.weights.shape[0])

        # Diverse initial voltages
        self.output_layer.V[:] = self.output_layer.V_rest + bm.random.normal(0, 2, n_output)

        # Lateral inhibition in hidden layer
        conn_hh = bp.conn.All2All(include_self=False)
        self.syn_hh = ExponentialSynapse(
            pre=self.hidden_layer,
            post=self.hidden_layer,
            conn=conn_hh,
            g_max=-0.5,
            tau=5.0
        )

        # Lateral inhibition in output layer (winner-take-all)
        conn_oo = bp.conn.All2All(include_self=False)
        self.syn_oo = ExponentialSynapse(
            pre=self.output_layer,
            post=self.output_layer,
            conn=conn_oo,
            g_max=-25.0,
            tau=3.0
        )

        # Inhibition shaped by distance (Marr wavelet-inspired)
        if hasattr(self.syn_oo, 'weights'):
            for i in range(n_output):
                for j in range(n_output):
                    if i != j:
                        dist = min(abs(i - j), n_output - abs(i - j))
                        if dist == 1:
                            self.syn_oo.weights[i, j] *= 1.5
                        elif dist == 2:
                            self.syn_oo.weights[i, j] *= 0.75
                        else:
                            self.syn_oo.weights[i, j] *= 0.25

    def update(self):
        # Update input layer (pass None to use internal rates)
        self.input_layer.update()
        
        # Update input to hidden connections first
        self.syn_ih.update()
        
        # Apply lateral inhibition in the hidden layer
        self.syn_hh.update()
        
        # Update hidden layer neurons
        self.hidden_layer.update()
        
        # Update hidden to output connections
        self.syn_ho.update()
        
        # Apply winner-take-all inhibition in the output layer
        self.syn_oo.update()
        
        # Update output layer neurons
        self.output_layer.update()
        
        return self.output_layer.spike.value

# Data Processing

def load_and_preprocess_digits():
    """Load digits dataset and preprocess for spiking network"""
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize pixel values to [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def encode_rate(data, max_rate=100.):
    """Convert normalized data to firing rates"""
    return data * max_rate

def decode_spikes(spike_counts, n_classes=10):
    """Return class with the highest spike count; break ties at random."""
    max_count = np.max(spike_counts)
    if max_count == 0:
        # no neuron fired – fall back to a random guess
        return np.random.randint(0, n_classes)

    winners = np.flatnonzero(spike_counts == max_count)
    if len(winners) == 1:
        return winners[0]          # clear winner
    return np.random.choice(winners)   # tie -> pick one at random

# Training and Evaluation
def run_classification_demo():
    """Run a demonstration of digit classification"""
    print("Loading digits dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_digits()
    
    print(f"Dataset shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Create network
    print("\nCreating spiking neural network...")
    net = SpikingDigitClassifier(
        n_input=64,
        n_output=10,
        use_stdp=False
    )

    # Simulation parameters
    dt = 0.1
    T_present = 100.0  # Time to present each sample
    n_samples = 20  # Number of samples to test
    
    # Select random test samples
    test_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Results storage
    predictions = []
    true_labels = []
    all_spike_data = []
    
    print(f"\nRunning inference on {n_samples} test samples...")
    
    for idx, test_idx in enumerate(test_indices):
        # Get sample
        sample = X_test[test_idx]
        label = y_test[test_idx]
                
        # Convert to firing rates
        input_rates = encode_rate(sample, max_rate=150.)
        
        # Debug: Check input rates
        if idx == 0:
            print(f"  Debug - Input rates min: {np.min(input_rates):.1f}, max: {np.max(input_rates):.1f}, mean: {np.mean(input_rates):.1f}")
        
        # Reset neuron states with added randomness to break symmetry
        net.hidden_layer.V[:] = net.hidden_layer.V_rest + bm.random.normal(0, 2.0, net.n_hidden)
        net.output_layer.V[:] = net.output_layer.V_rest + bm.random.normal(0, 2.0, net.n_output)
        
        net.hidden_layer.spike[:] = False
        net.output_layer.spike[:] = False
        
        net.hidden_layer.refractory[:] = 0.
        net.output_layer.refractory[:] = 0.
        
        # Reset adaptation variables and thresholds
        net.hidden_layer.adaptation[:] = bm.random.uniform(0, 0.1, net.n_hidden)
        net.output_layer.adaptation[:] = bm.random.uniform(0, 0.1, net.n_output)
        net.hidden_layer.V_th[:] = net.hidden_layer.V_th_base
        net.output_layer.V_th[:] = net.output_layer.V_th_base

        # Reset synaptic conductances (Crucial for preventing interference between samples)
        net.syn_ih.g[:] = 0.
        net.syn_ho.g[:] = 0.
        net.syn_hh.g[:] = 0.
        net.syn_oo.g[:] = 0.
        
        # Reset input layer states
        net.input_layer.spike[:] = False
        net.input_layer.rates[:] = input_rates

        # Create runner for this sample
        runner = bp.DSRunner(
            net,
            monitors={
                'input_spikes': net.input_layer.spike,
                'hidden_V': net.hidden_layer.V,
                'hidden_spikes': net.hidden_layer.spike,
                'output_V': net.output_layer.V,
                'output_spikes': net.output_layer.spike,
            },
            dt=dt,
            progress_bar=False
        )
        
        # Run simulation
        runner.run(T_present)
        
        # Count output spikes
        output_spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
        prediction = decode_spikes(output_spike_counts)
        
        predictions.append(prediction)
        true_labels.append(label)
        
        # Store spike data for visualization
        if idx < 5:  # Store first 5 for plotting
            all_spike_data.append({
                'input_spikes': runner.mon['input_spikes'],
                'hidden_spikes': runner.mon['hidden_spikes'],
                'output_spikes': runner.mon['output_spikes'],
                'output_V': runner.mon['output_V'],
                'times': runner.mon.ts,
                'true_label': label,
                'prediction': prediction,
                'sample': sample.reshape(8, 8)
            })
        
        print(f"Sample {idx+1}: True={label}, Predicted={prediction}, "
              f"Correct={'✓' if prediction == label else '✗'}")
        
        # Debug info for first sample
        if idx == 0:
            print(f"  Debug - Output spike counts: {output_spike_counts}")
            print(f"  Debug - Max output voltage: {np.max(runner.mon['output_V']):.2f}")
            print(f"  Debug - Input spike rate: {np.mean(np.sum(runner.mon['input_spikes'], axis=0)):.1f} spikes")
            print(f"  Debug - Hidden spike rate: {np.mean(np.sum(runner.mon['hidden_spikes'], axis=0)):.1f} spikes")
            total_input_spikes = np.sum(runner.mon['input_spikes'])
            print(f"  Debug - Total input spikes: {total_input_spikes}")
            if total_input_spikes == 0:
                print("  WARNING: No input spikes generated!")
            if len(np.unique(output_spike_counts)) <= 2:
                print("  WARNING: Output neurons have very similar spike counts!")
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Visualize results
    visualize_classification_results(all_spike_data)
    
    return net, predictions, true_labels

def visualize_classification_results(spike_data_list):
    """Visualize the classification results with better scaling"""
    n_samples = len(spike_data_list)
    
    # Adjust figure size based on number of samples
    fig_height = min(3 * n_samples, 15)  # Cap at 15 inches height
    fig = plt.figure(figsize=(16, fig_height))
    
    # Create a more compact grid
    gs = gridspec.GridSpec(n_samples, 4, 
                          width_ratios=[0.8, 2, 2, 2],
                          hspace=0.4,  # Add vertical spacing
                          wspace=0.3)  # Add horizontal spacing
    
    for i, data in enumerate(spike_data_list):
        # Input image - smaller
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(data['sample'], cmap='gray')
        ax_img.set_title(f"True: {data['true_label']}\nPred: {data['prediction']}", 
                         fontsize=10)
        ax_img.axis('off')
        
        # Input spikes
        ax_input = fig.add_subplot(gs[i, 1])
        spike_times, spike_neurons = np.where(data['input_spikes'])
        if len(spike_times) > 0:
            ax_input.scatter(data['times'][spike_times], spike_neurons, 
                           s=0.5, c='black', alpha=0.5, rasterized=True)
        ax_input.set_ylabel('Input', fontsize=9)
        ax_input.set_title('Input spikes', fontsize=10)
        ax_input.set_xlim(0, data['times'][-1])
        ax_input.tick_params(labelsize=8)
        
        # Hidden layer spikes
        ax_hidden = fig.add_subplot(gs[i, 2])
        spike_times, spike_neurons = np.where(data['hidden_spikes'])
        if len(spike_times) > 0:
            ax_hidden.scatter(data['times'][spike_times], spike_neurons, 
                            s=1, c='blue', alpha=0.6, rasterized=True)
        ax_hidden.set_ylabel('Hidden', fontsize=9)
        ax_hidden.set_title('Hidden layer spikes', fontsize=10)
        ax_hidden.set_xlim(0, data['times'][-1])
        ax_hidden.tick_params(labelsize=8)
        
        # Output layer - more compact
        ax_output = fig.add_subplot(gs[i, 3])
        
        # Plot voltage traces more compactly
        spacing = 15  # Reduced from 20
        for j in range(10):
            voltage = data['output_V'][:, j]
            ax_output.plot(data['times'], voltage + j*spacing, 'k-', 
                          linewidth=0.3, alpha=0.7)
            
            # Mark spikes
            spike_mask = data['output_spikes'][:, j]
            if np.any(spike_mask):
                spike_times = data['times'][spike_mask]
                spike_voltages = voltage[spike_mask] + j*spacing
                ax_output.scatter(spike_times, spike_voltages, 
                                c='red', s=15, marker='o', alpha=0.8)
        
        ax_output.set_ylabel('Output', fontsize=9)
        ax_output.set_title('Output layer (V + spikes)', fontsize=10)
        ax_output.set_ylim(-10, 150)  # Adjusted from 200
        ax_output.set_xlim(0, data['times'][-1])
        ax_output.tick_params(labelsize=8)
        
        # Add neuron labels
        ax_output.set_yticks([i*spacing for i in range(10)])
        ax_output.set_yticklabels([str(i) for i in range(10)], fontsize=8)
        
        # Only label x-axis on bottom row
        if i == n_samples - 1:
            ax_input.set_xlabel('Time (ms)', fontsize=9)
            ax_hidden.set_xlabel('Time (ms)', fontsize=9)
            ax_output.set_xlabel('Time (ms)', fontsize=9)
        else:
            ax_input.set_xticklabels([])
            ax_hidden.set_xticklabels([])
            ax_output.set_xticklabels([])
    
    #plt.tight_layout()
    
    plt.show()

def analyze_network_activity(net, X_test, y_test, n_samples=5):
    """Analyze network activity patterns with better layout"""
    dt = 0.1
    T_present = 200.0
    
    # Select samples from different classes
    class_samples = {}
    for cls in range(10):
        indices = np.where(y_test == cls)[0]
        if len(indices) > 0:
            class_samples[cls] = X_test[indices[0]]
    
    # Create figure with better size
    fig = plt.figure(figsize=(12, 8))
    
    # Use GridSpec for better control
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Analyze firing rates per class
    ax = fig.add_subplot(gs[0, 0])
    class_rates = {}
    
    for cls, sample in class_samples.items():
        input_rates = encode_rate(sample, max_rate=150.)
        
        runner = bp.DSRunner(
            net,
            monitors={'output_spikes': net.output_layer.spike},
            dt=dt,
            progress_bar=False
        )
        
        net.input_layer.rates[:] = input_rates
        runner.run(T_present)
        
        spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
        firing_rates = spike_counts / (T_present / 1000)
        class_rates[cls] = firing_rates
    
    # Plot firing rate matrix
    rate_matrix = np.array([class_rates.get(i, np.zeros(10)) for i in range(10)])
    im = ax.imshow(rate_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Output neuron', fontsize=10)
    ax.set_ylabel('Input class', fontsize=10)
    ax.set_title('Output firing rates by input class', fontsize=11)
    ax.tick_params(labelsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Firing rate (Hz)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Weight distribution
    ax = fig.add_subplot(gs[0, 1])
    if hasattr(net.syn_ih, 'w'):
        weights = net.syn_ih.w.value.flatten()
    elif hasattr(net.syn_ih, 'weights'):
        weights = net.syn_ih.weights.value.flatten()
    
    weights = weights[weights > 0]  # Only connected weights
    ax.hist(weights, bins=30, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Synaptic weight', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Input-Hidden weight distribution', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    
    # Population activity over time
    ax = fig.add_subplot(gs[1, 0])
    sample = X_test[0]
    input_rates = encode_rate(sample, max_rate=150.)
    
    runner = bp.DSRunner(
        net,
        monitors={
            'hidden_spikes': net.hidden_layer.spike,
            'output_spikes': net.output_layer.spike
        },
        dt=dt,
        progress_bar=False
    )
    
    net.input_layer.rates[:] = input_rates
    runner.run(T_present)
    
    # Calculate population rates with smoothing
    window = 10  # ms
    window_steps = int(window / dt)
    times = runner.mon.ts
    
    hidden_pop_rate = np.convolve(
        np.mean(runner.mon['hidden_spikes'], axis=1),
        np.ones(window_steps)/window_steps,
        mode='same'
    ) * 1000 / dt
    
    output_pop_rate = np.convolve(
        np.mean(runner.mon['output_spikes'], axis=1),
        np.ones(window_steps)/window_steps,
        mode='same'
    ) * 1000 / dt
    
    ax.plot(times, hidden_pop_rate, label='Hidden layer', alpha=0.8, linewidth=1.5)
    ax.plot(times, output_pop_rate, label='Output layer', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Population firing rate (Hz)', fontsize=10)
    ax.set_title('Population activity dynamics', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Spike count distribution
    ax = fig.add_subplot(gs[1, 1])
    all_hidden_counts = []
    all_output_counts = []
    
    for i in range(min(20, len(X_test))):
        sample = X_test[i]
        input_rates = encode_rate(sample, max_rate=150.)
        
        runner = bp.DSRunner(
            net,
            monitors={
                'hidden_spikes': net.hidden_layer.spike,
                'output_spikes': net.output_layer.spike
            },
            dt=dt,
            progress_bar=False
        )
        
        net.input_layer.rates[:] = input_rates
        runner.run(T_present)
        
        all_hidden_counts.extend(np.sum(runner.mon['hidden_spikes'], axis=0))
        all_output_counts.extend(np.sum(runner.mon['output_spikes'], axis=0))
    
    # Create histograms with better styling
    bins = np.linspace(0, max(max(all_hidden_counts), max(all_output_counts)), 20)
    ax.hist(all_hidden_counts, bins=bins, alpha=0.5, label='Hidden', 
            density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax.hist(all_output_counts, bins=bins, alpha=0.5, label='Output', 
            density=True, color='red', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Spike count per neuron', fontsize=10)
    ax.set_ylabel('Probability density', fontsize=10)
    ax.set_title('Spike count distribution', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle('Network Activity Analysis', fontsize=14, y=0.98)
    
    #plt.tight_layout()
    plt.show()

def test_stdp_learning():
    """Test STDP learning effectiveness with model saving"""
    print("\n" + "="*50)
    print("Testing STDP Learning with Save/Load")
    print("="*50)
    
    # Check for existing models
    model_file = prompt_user_for_model()
    
    if model_file is not None:
        # Load existing model
        print(f"\nLoading model: {model_file}")
        net_stdp = load_network_weights(model_file, use_stdp=True)
        
        # Ask if user wants to continue training
        continue_training = input("\nContinue training this model? (y/n): ").lower() == 'y'
    else:
        # Create new model
        print("\nCreating new STDP network...")
        net_stdp = SpikingDigitClassifier(
            n_input=64,
            n_output=10,
            use_stdp=True
        )
        continue_training = True
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_digits()
    
    def evaluate_network(net, X_eval, y_eval, n_samples=20):
        """Evaluate network accuracy"""
        correct = 0
        dt = 0.1
        T_present = 100.0
        
        for i in range(min(n_samples, len(X_eval))):
            sample = X_eval[i]
            label = y_eval[i]
            input_rates = encode_rate(sample, max_rate=150.)
            
            # Reset network
            net.input_layer.rates[:] = input_rates
            
            runner = bp.DSRunner(
                net,
                monitors={'output_spikes': net.output_layer.spike},
                dt=dt,
                progress_bar=False
            )
            
            runner.run(T_present)
            
            # Get prediction
            output_spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
            prediction = decode_spikes(output_spike_counts)
            
            if prediction == label:
                correct += 1
                
        return correct / n_samples
    
    # Initial performance
    print("\nEvaluating current performance...")
    init_acc = evaluate_network(net_stdp, X_test, y_test, 50)
    print(f"Current accuracy: {init_acc:.2%}")
    
    if continue_training:
        # Store initial weights for comparison
        if hasattr(net_stdp.syn_ih, 'w'):
            initial_weights = net_stdp.syn_ih.w.value.copy()
        
        # Training phase
        print("\nTraining with STDP...")
        n_epochs = int(input("Number of training epochs (default 5): ") or "5")
        n_train_samples = int(input("Number of training samples per epoch (default 100): ") or "100")
        dt = 0.1
        T_present = 200.0  # Longer presentation for learning
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            epoch_accuracy = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))[:n_train_samples]
            
            for idx, i in enumerate(indices):
                sample = X_train[i]
                label = y_train[i]
                input_rates = encode_rate(sample, max_rate=150.)
                
                # Set input
                net_stdp.input_layer.rates[:] = input_rates
                
                # Run simulation (weights update during this)
                runner = bp.DSRunner(
                    net_stdp,
                    monitors={'output_spikes': net_stdp.output_layer.spike},
                    dt=dt,
                    progress_bar=False
                )
                
                runner.run(T_present)
                
                # Check if correct
                output_spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
                prediction = decode_spikes(output_spike_counts)
                if prediction == label:
                    epoch_accuracy += 1
                
                if (idx + 1) % 20 == 0:
                    print(f"  Processed {idx+1}/{n_train_samples} samples, "
                          f"running accuracy: {epoch_accuracy/(idx+1):.2%}")
            
            # Evaluate after each epoch
            test_acc = evaluate_network(net_stdp, X_test, y_test, 50)
            print(f"  Test accuracy after epoch {epoch+1}: {test_acc:.2%}")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_acc = evaluate_network(net_stdp, X_test, y_test, 100)
        
        print(f"\nResults:")
        print(f"Accuracy: {init_acc:.2%} -> {final_acc:.2%} "
              f"({final_acc - init_acc:+.2%} improvement)")
        
        # Analyze weight changes
        if hasattr(net_stdp.syn_ih, 'w'):
            final_weights = net_stdp.syn_ih.w.value
            weight_change = np.mean(np.abs(final_weights - initial_weights))
            print(f"Average weight change: {weight_change:.4f}")
            
            # Visualize weight changes
            visualize_stdp_weights(initial_weights, final_weights)
    else:
        final_acc = init_acc
    
    # Save model
    save_choice = input("\nSave this model? (y/n): ").lower()
    if save_choice == 'y':
        custom_name = input("Enter filename (press Enter for auto-generated): ").strip()
        if custom_name and not custom_name.endswith('.pkl'):
            custom_name += '.pkl'
        filename = save_network_weights(net_stdp, custom_name if custom_name else None)
        
        # Optional: delete old models
        if len(list_saved_models()) > 5:
            delete_old = input("\nMore than 5 models saved. Delete oldest models? (y/n): ").lower()
            if delete_old == 'y':
                old_models = list_saved_models()[:-5]  # Keep only 5 most recent
                for old_model in old_models:
                    os.remove(old_model)
                    print(f"Deleted: {old_model}")
    
    return net_stdp, final_acc

def visualize_stdp_weights(initial_weights, final_weights):
    """Visualize STDP weight changes with better layout"""
    # Create figure with appropriate size
    fig = plt.figure(figsize=(14, 5))
    
    # Calculate weight changes
    weight_changes = final_weights - initial_weights
    
    # Determine color scale limits
    vmax = max(np.max(initial_weights), np.max(final_weights))
    vmin = 0
    change_limit = max(abs(np.min(weight_changes)), np.max(weight_changes))
    
    # Initial weights
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(initial_weights[:, :30], aspect='auto', cmap='hot',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title('Initial Weights (first 30 hidden)', fontsize=12)
    ax1.set_xlabel('Hidden neurons', fontsize=10)
    ax1.set_ylabel('Input neurons', fontsize=10)
    ax1.tick_params(labelsize=9)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    
    # Final weights
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(final_weights[:, :30], aspect='auto', cmap='hot',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_title('Final Weights (first 30 hidden)', fontsize=12)
    ax2.set_xlabel('Hidden neurons', fontsize=10)
    ax1.set_ylabel('Input neurons', fontsize=10)
    ax1.tick_params(labelsize=9)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    
    # Final weights
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(final_weights[:, :30], aspect='auto', cmap='hot',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_title('Final Weights (first 30 hidden)', fontsize=12)
    ax2.set_xlabel('Hidden neurons', fontsize=10)
    ax2.set_ylabel('Input neurons', fontsize=10)
    ax2.tick_params(labelsize=9)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=8)
    
    # Weight changes
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(weight_changes[:, :30], aspect='auto', cmap='RdBu_r',
                     vmin=-change_limit, vmax=change_limit, interpolation='nearest')
    ax3.set_title('Weight Changes (LTP/LTD)', fontsize=12)
    ax3.set_xlabel('Hidden neurons', fontsize=10)
    ax3.set_ylabel('Input neurons', fontsize=10)
    ax3.tick_params(labelsize=9)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=8)
    cbar3.set_label('ΔW', fontsize=9)
    
    # Add overall statistics
    fig.text(0.5, 0.02, f'Mean weight change: {np.mean(np.abs(weight_changes)):.4f}', 
             ha='center', fontsize=10)
    
    #plt.tight_layout()
    plt.show()

# Main execution

if __name__ == "__main__":
    print("Extended LIF Network with Digit Classification")
    print("=" * 50)
    
    # Main menu
    print("\nWhat would you like to do?")
    print("1. Run classification demo")
    print("2. Train/Load STDP model")
    print("3. Analyze existing model")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        # Run classification demo
        net, predictions, true_labels = run_classification_demo()
        
        # Analyze network activity
        print("\nAnalyzing network activity patterns...")
        X_train, X_test, y_train, y_test = load_and_preprocess_digits()
        analyze_network_activity(net, X_test, y_test)
        
    elif choice == '2':
        # Train or load STDP model
        net_stdp, accuracy = test_stdp_learning()
        
        # Optional: run additional analysis
        if input("\nAnalyze network activity? (y/n): ").lower() == 'y':
            X_train, X_test, y_train, y_test = load_and_preprocess_digits()
            analyze_network_activity(net_stdp, X_test, y_test)
            
    elif choice == '3':
        # Load and analyze existing model
        model_file = prompt_user_for_model()
        if model_file is not None:
            print(f"\nLoading model: {model_file}")
            net = load_network_weights(model_file, use_stdp=False)
            
            # Run analysis
            X_train, X_test, y_train, y_test = load_and_preprocess_digits()
            
            # Quick accuracy check
            print("\nEvaluating loaded model...")
            dt = 0.1
            T_present = 100.0
            correct = 0
            n_test = 50
            
            for i in range(n_test):
                sample = X_test[i]
                label = y_test[i]
                input_rates = encode_rate(sample, max_rate=150.)
                
                net.input_layer.rates[:] = input_rates
                
                runner = bp.DSRunner(
                    net,
                    monitors={'output_spikes': net.output_layer.spike},
                    dt=dt,
                    progress_bar=False
                )
                
                runner.run(T_present)
                
                output_spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
                prediction = decode_spikes(output_spike_counts)
                
                if prediction == label:
                    correct += 1
            
            print(f"Accuracy: {correct/n_test:.2%}")
            
            # Detailed analysis
            analyze_network_activity(net, X_test, y_test)
        else:
            print("No model selected.")
    else:
        print("Invalid choice.")
