import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec

bp.math.set_platform('cpu')

# ============================================================================
# Neuron Models
# ============================================================================

class LIFNeuron(bp.dyn.NeuDyn):
    """Leaky Integrate-and-Fire neuron with adaptive threshold"""
    def __init__(self, size, V_rest=-65., V_th=-50., V_reset=-65., tau=20., 
                 tau_ref=5., tau_adapt=100., a_adapt=0.01):
        super().__init__(size=size)
        
        # Parameters
        self.V_rest = V_rest
        self.V_th_base = V_th
        self.V_reset = V_reset
        self.tau = tau
        self.tau_ref = tau_ref
        self.tau_adapt = tau_adapt
        self.a_adapt = a_adapt
        
        # Variables
        self.V = bm.Variable(bm.ones(size) * V_rest)
        self.V_th = bm.Variable(bm.ones(size) * V_th)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.refractory = bm.Variable(bm.zeros(size))
        self.input = bm.Variable(bm.zeros(size))
        self.adaptation = bm.Variable(bm.zeros(size))
        
    def update(self, inp=None):
        # Get total input
        total_input = self.input.value
        if inp is not None:
            total_input = total_input + inp
        
        # Add small noise to break symmetry
        noise = bm.random.normal(0, 0.5, self.num)
        total_input = total_input + noise
            
        # Update refractory period
        self.refractory.value = bm.maximum(self.refractory - 1, 0)
        
        # Only update neurons not in refractory period
        not_refractory = self.refractory <= 0
        
        # Update membrane potential for non-refractory neurons
        dV = (self.V_rest - self.V + total_input) / self.tau
        V_new = bm.where(not_refractory, self.V + dV, self.V_reset)
        
        # Update adaptation
        self.adaptation.value -= self.adaptation / self.tau_adapt
        
        # Update threshold with adaptation
        self.V_th.value = self.V_th_base + self.adaptation
        
        # Check for spikes
        self.spike.value = bm.logical_and(V_new >= self.V_th, not_refractory)
        
        # Reset neurons that spiked
        V_new = bm.where(self.spike, self.V_reset, V_new)
        
        # Update adaptation for spiking neurons
        self.adaptation.value = bm.where(self.spike, 
                                        self.adaptation + self.a_adapt, 
                                        self.adaptation)
        
        # Set refractory period for neurons that spiked
        self.refractory.value = bm.where(self.spike, self.tau_ref, self.refractory)
        
        # Update voltage
        self.V.value = V_new
        
        # Clear input
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
            self.rates[:] = rates  # Use assignment to existing variable
        # Generate spikes based on Poisson process
        prob = self.rates * 0.001 * 1.5  # Increased probability to generate more spikes
        # Fix: use self.num instead of self.size for random generation
        self.spike.value = bm.random.rand(self.num) < prob
        return self.spike.value


# ============================================================================
# Synapse Models
# ============================================================================

class ExponentialSynapse(bp.dyn.SynConn):
    """Exponential synapse with synaptic delay"""
    def __init__(self, pre, post, conn, g_max=1., tau=10., delay=1.):
        super().__init__(pre=pre, post=post, conn=conn)
        
        # Parameters
        self.g_max = g_max
        self.tau = tau
        self.delay = int(delay)
        
        # Get connection matrix
        self.conn_mat = self.conn.require('conn_mat')
        
        # Initialize weights with some randomness if excitatory
        if g_max > 0:
            # Random weights between 0.5 and 1.0 times g_max
            self.weights = bm.Variable(
                self.conn_mat * bm.random.uniform(0.5, 1.0, self.conn_mat.shape) * g_max
            )
        else:
            # For inhibitory, use fixed strength
            self.weights = bm.Variable(self.conn_mat * g_max)
        
        # Synaptic conductance
        self.g = bm.Variable(bm.zeros(post.num))
        
        # Spike history buffer for delays
        self.spike_buffer = bm.Variable(bm.zeros((self.delay, pre.num), dtype=bool))
        
    def update(self):
        # Update synaptic conductance with exponential decay
        self.g.value = self.g.value * (1 - 1/self.tau)
        
        # Get delayed spikes
        if self.delay > 1:
            delayed_spikes = self.spike_buffer[0]
            # Shift buffer
            self.spike_buffer[:-1] = self.spike_buffer[1:]
            self.spike_buffer[-1] = self.pre.spike
        else:
            delayed_spikes = self.pre.spike
        
        # Add input from pre-synaptic spikes
        spike_input = bm.dot(delayed_spikes.astype(bm.float32), 
                           self.weights.astype(bm.float32))
        self.g.value = self.g.value + spike_input
        
        # Apply conductance to post-synaptic input
        self.post.input.value = self.post.input.value + self.g.value


class STDPSynapse(bp.dyn.SynConn):
    """Spike-Timing Dependent Plasticity synapse"""
    def __init__(self, pre, post, conn, g_max=1., tau=10., 
                 tau_pre=20., tau_post=20., A_pre=0.01, A_post=0.01,
                 w_min=0., w_max=2.):
        super().__init__(pre=pre, post=post, conn=conn)
        
        # Parameters
        self.tau = tau
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_pre = A_pre
        self.A_post = A_post
        self.w_min = w_min
        self.w_max = w_max
        
        # Get connection matrix shape
        self.conn_mat = self.conn.require('conn_mat')
        
        # Synaptic weights (plastic) - initialize with random values
        self.w = bm.Variable(
            self.conn_mat * bm.random.uniform(0.5 * g_max, g_max, self.conn_mat.shape)
        )
        
        # Synaptic conductance
        self.g = bm.Variable(bm.zeros(post.num))
        
        # STDP traces
        self.trace_pre = bm.Variable(bm.zeros(pre.num))
        self.trace_post = bm.Variable(bm.zeros(post.num))
        
    def update(self):
        # Update synaptic conductance
        self.g.value = self.g.value * (1 - 1/self.tau)
        
        # Add input from pre-synaptic spikes
        spike_input = bm.dot(self.pre.spike.astype(bm.float32), 
                           (self.conn_mat * self.w).astype(bm.float32))
        self.g.value = self.g.value + spike_input
        
        # Update STDP traces
        self.trace_pre.value *= (1 - 1/self.tau_pre)
        self.trace_post.value *= (1 - 1/self.tau_post)
        
        # Update traces for spiking neurons
        self.trace_pre.value = bm.where(self.pre.spike, 
                                       self.trace_pre + 1.0, 
                                       self.trace_pre)
        self.trace_post.value = bm.where(self.post.spike, 
                                        self.trace_post + 1.0, 
                                        self.trace_post)
        
        # STDP weight updates using vectorized operations
        # LTD: When pre-synaptic neuron spikes
        pre_spike_expanded = self.pre.spike[:, None].astype(bm.float32)
        post_trace_expanded = self.trace_post[None, :].astype(bm.float32)
        ltd_update = -self.A_post * pre_spike_expanded * post_trace_expanded * self.w
        
        # LTP: When post-synaptic neuron spikes
        pre_trace_expanded = self.trace_pre[:, None].astype(bm.float32)
        post_spike_expanded = self.post.spike[None, :].astype(bm.float32)
        ltp_update = self.A_pre * pre_trace_expanded * post_spike_expanded * (self.w_max - self.w)
        
        # Apply updates only where connections exist
        weight_update = (ltd_update + ltp_update) * self.conn_mat
        self.w.value = self.w + weight_update
        
        # Clip weights
        self.w.value = bm.clip(self.w, self.w_min, self.w_max)
        
        # Apply conductance to post-synaptic input
        self.post.input.value = self.post.input.value + self.g.value


# ============================================================================
# Network Model
# ============================================================================

class SpikingDigitClassifier(bp.dyn.Network):
    """Multi-layer spiking neural network for digit classification"""
    def __init__(self, n_input=64, n_hidden=100, n_output=10, 
                 use_stdp=False, connection_prob=0.5):
        super().__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Input layer (Poisson neurons)
        self.input_layer = PoissonInput(size=n_input, freq_max=200.)  # Increased max frequency
        
        # Hidden layer
        self.hidden_layer = LIFNeuron(
            size=n_hidden, 
            V_rest=-65., 
            V_th=-52.,  # Lower threshold for easier firing
            V_reset=-65., 
            tau=20., 
            tau_ref=3.,  # Shorter refractory period
            tau_adapt=100.,
            a_adapt=0.005  # Less adaptation
        )
        
        # Output layer
        self.output_layer = LIFNeuron(
            size=n_output, 
            V_rest=-65., 
            V_th=-52.,  # Lower threshold for easier firing
            V_reset=-65., 
            tau=15., 
            tau_ref=3.,  # Shorter refractory period
            tau_adapt=50.,
            a_adapt=0.005  # Less adaptation
        )
        
        # Connections
        # Input to Hidden: random connectivity
        conn_ih = bp.conn.FixedProb(prob=connection_prob)
        if use_stdp:
            self.syn_ih = STDPSynapse(
                pre=self.input_layer,
                post=self.hidden_layer,
                conn=conn_ih,
                g_max=25.0,  # Increased initial weight
                tau=10.0,
                A_pre=0.01,
                A_post=0.01
            )
        else:
            self.syn_ih = ExponentialSynapse(
                pre=self.input_layer,
                post=self.hidden_layer,
                conn=conn_ih,
                g_max=25.0,  # Increased synaptic strength
                tau=10.0
            )
        
        # Hidden to Output: all-to-all connectivity with variation
        conn_ho = bp.conn.All2All()
        self.syn_ho = ExponentialSynapse(
            pre=self.hidden_layer,
            post=self.output_layer,
            conn=conn_ho,
            g_max=20.0,  # Increased synaptic strength
            tau=15.0
        )
        
        # Add output-specific bias by slightly modifying weights
        # This helps different output neurons specialize
        for i in range(n_output):
            # Add small bias to weights going to each output neuron
            self.syn_ho.weights[:, i] *= (1.0 + 0.1 * (i - n_output/2) / n_output)
        
        # Lateral inhibition in hidden layer
        conn_hh = bp.conn.All2All(include_self=False)
        self.syn_hh = ExponentialSynapse(
            pre=self.hidden_layer,
            post=self.hidden_layer,
            conn=conn_hh,
            g_max=-1.0,  # Reduced inhibition
            tau=5.0
        )
        
        # Lateral inhibition in output layer (winner-take-all)
        conn_oo = bp.conn.All2All(include_self=False)
        self.syn_oo = ExponentialSynapse(
            pre=self.output_layer,
            post=self.output_layer,
            conn=conn_oo,
            g_max=-3.0,  # Reduced inhibition
            tau=5.0
        )
        
    def update(self):
        # Update input layer with stored rates (pass None to use internal rates)
        self.input_layer.update()
        
        # Update input to hidden connections first
        self.syn_ih.update()
        
        # Update hidden layer neurons
        self.hidden_layer.update()
        
        # Apply lateral inhibition after neurons have fired
        self.syn_hh.update()
        
        # Update hidden to output connections
        self.syn_ho.update()
        
        # Update output layer neurons
        self.output_layer.update()
        
        # Apply lateral inhibition in output layer after firing
        self.syn_oo.update()
        
        return self.output_layer.spike.value


# ============================================================================
# Data Processing
# ============================================================================

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
    """Decode class from spike counts"""
    # If no spikes, return random guess to avoid always predicting 0
    if np.sum(spike_counts) == 0:
        return np.random.randint(0, n_classes)
    return np.argmax(spike_counts)


# ============================================================================
# Training and Evaluation
# ============================================================================

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
        n_hidden=100,
        n_output=10,
        use_stdp=False,
        connection_prob=0.5  # Increased connection probability
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
        input_rates = encode_rate(sample, max_rate=150.)  # Increased max rate
        
        # Debug: Check input rates
        if idx == 0:
            print(f"  Debug - Input rates min: {np.min(input_rates):.1f}, max: {np.max(input_rates):.1f}, mean: {np.mean(input_rates):.1f}")
        
        # Reset network state
        net.input_layer.spike[:] = False
        net.hidden_layer.V[:] = net.hidden_layer.V_rest
        net.hidden_layer.spike[:] = False
        net.hidden_layer.refractory[:] = 0
        net.output_layer.V[:] = net.output_layer.V_rest
        net.output_layer.spike[:] = False
        net.output_layer.refractory[:] = 0
        
        # Reset synaptic conductances
        net.syn_ih.g[:] = 0.0
        net.syn_ho.g[:] = 0.0
        net.syn_hh.g[:] = 0.0
        net.syn_oo.g[:] = 0.0
        
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
            progress_bar=False  # Disable progress bar for cleaner output
        )
        
        # Set input rates
        net.input_layer.rates[:] = input_rates
        net.input_layer.rates[:] = input_rates  # Also set directly in input layer
        net.input_layer.rates[:] = input_rates  # Also set directly in input layer
        
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
            # Check if input is actually spiking
            total_input_spikes = np.sum(runner.mon['input_spikes'])
            print(f"  Debug - Total input spikes: {total_input_spikes}")
            if total_input_spikes == 0:
                print("  WARNING: No input spikes generated!")
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Visualize results
    visualize_classification_results(all_spike_data)
    
    return net, predictions, true_labels


def visualize_classification_results(spike_data_list):
    """Visualize the classification results"""
    n_samples = len(spike_data_list)
    
    fig = plt.figure(figsize=(20, 4*n_samples))
    gs = gridspec.GridSpec(n_samples, 4, width_ratios=[1, 2, 2, 2])
    
    for i, data in enumerate(spike_data_list):
        # Input image
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(data['sample'], cmap='gray')
        ax_img.set_title(f"True: {data['true_label']}\nPred: {data['prediction']}")
        ax_img.axis('off')
        
        # Input spikes
        ax_input = fig.add_subplot(gs[i, 1])
        spike_times, spike_neurons = np.where(data['input_spikes'])
        if len(spike_times) > 0:
            ax_input.scatter(data['times'][spike_times], spike_neurons, 
                           s=1, c='black', alpha=0.5)
        ax_input.set_ylabel('Input neuron')
        ax_input.set_title('Input spikes')
        ax_input.set_xlim(0, data['times'][-1])
        
        # Hidden layer spikes
        ax_hidden = fig.add_subplot(gs[i, 2])
        spike_times, spike_neurons = np.where(data['hidden_spikes'])
        if len(spike_times) > 0:
            ax_hidden.scatter(data['times'][spike_times], spike_neurons, 
                            s=2, c='blue', alpha=0.6)
        ax_hidden.set_ylabel('Hidden neuron')
        ax_hidden.set_title('Hidden layer spikes')
        ax_hidden.set_xlim(0, data['times'][-1])
        
        # Output layer
        ax_output = fig.add_subplot(gs[i, 3])
        for j in range(10):
            voltage = data['output_V'][:, j]
            ax_output.plot(data['times'], voltage + j*20, 'k-', linewidth=0.5)
            
            # Mark spikes
            spike_mask = data['output_spikes'][:, j]
            if np.any(spike_mask):
                spike_times = data['times'][spike_mask]
                spike_voltages = voltage[spike_mask] + j*20
                ax_output.scatter(spike_times, spike_voltages, 
                                c='red', s=20, marker='o')
        
        ax_output.set_ylabel('Output neurons')
        ax_output.set_title('Output layer (voltage + spikes)')
        ax_output.set_ylim(-10, 200)
        ax_output.set_xlim(0, data['times'][-1])
        
        # Add neuron labels
        ax_output.set_yticks([i*20 for i in range(10)])
        ax_output.set_yticklabels([str(i) for i in range(10)])
        
        if i == n_samples - 1:
            ax_input.set_xlabel('Time (ms)')
            ax_hidden.set_xlabel('Time (ms)')
            ax_output.set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()


def analyze_network_activity(net, X_test, y_test, n_samples=5):
    """Analyze network activity patterns"""
    dt = 0.1
    T_present = 200.0
    
    # Select samples from different classes
    class_samples = {}
    for cls in range(10):
        indices = np.where(y_test == cls)[0]
        if len(indices) > 0:
            class_samples[cls] = X_test[indices[0]]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Analyze firing rates per class
    ax = axes[0, 0]
    class_rates = {}
    
    for cls, sample in class_samples.items():
        input_rates = encode_rate(sample, max_rate=150.)  # Increased max rate
        
        runner = bp.DSRunner(
            net,
            monitors={'output_spikes': net.output_layer.spike},
            dt=dt,
            progress_bar=False
        )
        
        # Set input rates
        net.input_rates[:] = input_rates
        
        runner.run(T_present)
        
        # Calculate firing rates for each output neuron
        spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
        firing_rates = spike_counts / (T_present / 1000)  # Hz
        class_rates[cls] = firing_rates
    
    # Plot firing rate matrix
    rate_matrix = np.array([class_rates.get(i, np.zeros(10)) for i in range(10)])
    im = ax.imshow(rate_matrix, aspect='auto', cmap='hot')
    ax.set_xlabel('Output neuron')
    ax.set_ylabel('Input class')
    ax.set_title('Output firing rates by input class')
    plt.colorbar(im, ax=ax, label='Firing rate (Hz)')
    
    # Weight distribution
    ax = axes[0, 1]
    if hasattr(net.syn_ih, 'w'):
        weights = net.syn_ih.w.value.flatten()
        weights = weights[weights > 0]  # Only connected weights
        ax.hist(weights, bins=30, alpha=0.7, color='blue')
        ax.set_xlabel('Synaptic weight')
        ax.set_ylabel('Count')
        ax.set_title('Input-Hidden weight distribution')
    elif hasattr(net.syn_ih, 'weights'):
        weights = net.syn_ih.weights.value.flatten()
        weights = weights[weights > 0]  # Only connected weights
        ax.hist(weights, bins=30, alpha=0.7, color='blue')
        ax.set_xlabel('Synaptic weight')
        ax.set_ylabel('Count')
        ax.set_title('Input-Hidden weight distribution')
    
    # Population activity over time
    ax = axes[1, 0]
    sample = X_test[0]
    input_rates = encode_rate(sample, max_rate=150.)  # Increased max rate
    
    runner = bp.DSRunner(
        net,
        monitors={
            'hidden_spikes': net.hidden_layer.spike,
            'output_spikes': net.output_layer.spike
        },
        dt=dt,
        progress_bar=False
    )
    
    # Set input rates
    net.input_layer.rates[:] = input_rates
    
    runner.run(T_present)
    
    # Calculate population rates
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
    
    ax.plot(times, hidden_pop_rate, label='Hidden layer', alpha=0.7)
    ax.plot(times, output_pop_rate, label='Output layer', alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Population firing rate (Hz)')
    ax.set_title('Population activity dynamics')
    ax.legend()
    
    # Spike count distribution
    ax = axes[1, 1]
    all_hidden_counts = []
    all_output_counts = []
    
    for i in range(min(20, len(X_test))):
        sample = X_test[i]
        input_rates = encode_rate(sample, max_rate=150.)  # Increased max rate
        
        runner = bp.DSRunner(
            net,
            monitors={
                'hidden_spikes': net.hidden_layer.spike,
                'output_spikes': net.output_layer.spike
            },
            dt=dt,
            progress_bar=False
        )
        
        # Set input rates
        net.input_rates[:] = input_rates
        
        runner.run(T_present)
        
        all_hidden_counts.extend(np.sum(runner.mon['hidden_spikes'], axis=0))
        all_output_counts.extend(np.sum(runner.mon['output_spikes'], axis=0))
    
    ax.hist(all_hidden_counts, bins=20, alpha=0.5, label='Hidden', density=True)
    ax.hist(all_output_counts, bins=20, alpha=0.5, label='Output', density=True)
    ax.set_xlabel('Spike count per neuron')
    ax.set_ylabel('Probability density')
    ax.set_title('Spike count distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Extended LIF Network with Digit Classification")
    print("=" * 50)
    
    # Run classification demo
    net, predictions, true_labels = run_classification_demo()
    
    # Analyze network activity
    #print("\nAnalyzing network activity patterns...")
    X_train, X_test, y_train, y_test = load_and_preprocess_digits()
    analyze_network_activity(net, X_test, y_test)
    
    # Additional experiments
    print("\nRunning additional experiments...")
    
    # Test with STDP
    print("\nTesting with STDP learning...")
    net_stdp = SpikingDigitClassifier(
        n_input=64,
        n_hidden=100,
        n_output=10,
        use_stdp=True,
        connection_prob=0.5  # Increased connection probability
    )
    
    # Simple training loop with STDP
    print("Training with STDP (simplified demonstration)...")
    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        # Train on subset of data
        for i in range(min(50, len(X_train))):
            sample = X_train[i]
            label = y_train[i]
            input_rates = encode_rate(sample, max_rate=150.)  # Increased max rate
            
            runner = bp.DSRunner(
                net_stdp,
                dt=0.1,
                progress_bar=False
            )
            
            # Set input rates
            net_stdp.input_layer.rates[:] = input_rates
            
            runner.run(50.0)  # Short presentation for training
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1} samples")
    
    print("\nExtended LIF network demonstration complete!")
