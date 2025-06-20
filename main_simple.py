import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import pickle
import os
from datetime import datetime

bp.math.set_platform('cpu')

# Simple Dataset Generation

def generate_simple_dataset(dataset_type='circles', n_samples=200, noise=0.1):
    """Generate simple 2D datasets for classification"""
    if dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'blobs':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                  n_informative=2, n_clusters_per_class=1, n_classes=2,
                                  random_state=42, class_sep=2.0)
    elif dataset_type == 'spiral':
        # Create a simple spiral dataset
        n_points = n_samples // 2
        theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        x_a = r_a * np.cos(theta) + np.random.randn(n_points) * noise
        y_a = r_a * np.sin(theta) + np.random.randn(n_points) * noise
        
        theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi
        r_b = -2 * theta - np.pi
        x_b = r_b * np.cos(theta) + np.random.randn(n_points) * noise
        y_b = r_b * np.sin(theta) + np.random.randn(n_points) * noise
        
        X = np.vstack([np.column_stack([x_a, y_a]), np.column_stack([x_b, y_b])])
        y = np.hstack([np.zeros(n_points), np.ones(n_points)])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Simple Neuron Models

class SimpleLIFNeuron(bp.dyn.NeuDyn):
    """Simplified Leaky Integrate-and-Fire neuron"""
    def __init__(self, size, V_rest=-65., V_th=-50., V_reset=-65., tau=20., 
                 tau_ref=5., noise_scale=0.5):
        super().__init__(size=size)
        
        # Parameters
        self.V_rest = V_rest
        self.V_th = V_th
        self.V_reset = V_reset
        self.tau = tau
        self.tau_ref = tau_ref
        self.noise_scale = noise_scale
        
        # Variables
        self.V = bm.Variable(bm.ones(size) * V_rest)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.refractory = bm.Variable(bm.zeros(size))
        self.input_current = bm.Variable(bm.zeros(size))
        
    def update(self):
        # Update refractory period
        self.refractory.value = bm.maximum(self.refractory - bm.dt, 0.)
        not_refractory = self.refractory <= 0.
        
        # Add noise to break symmetry
        noise = bm.random.normal(0., self.noise_scale, self.num)
        
        # Membrane potential dynamics
        dV = (self.V_rest - self.V + self.input_current + noise) / self.tau
        self.V.value = bm.where(not_refractory, self.V + dV * bm.dt, self.V)
        
        # Spike generation
        self.spike.value = bm.logical_and(self.V >= self.V_th, not_refractory)
        
        # Reset
        self.V.value = bm.where(self.spike, self.V_reset, self.V)
        self.refractory.value = bm.where(self.spike, self.tau_ref, self.refractory)
        
        # Clear input
        self.input_current[:] = 0.
        
        return self.spike.value

class SimpleInputNeuron(bp.dyn.NeuDyn):
    """Simple rate-based input neuron"""
    def __init__(self, size):
        super().__init__(size=size)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.rates = bm.Variable(bm.zeros(size))
        
    def update(self):
        # Generate spikes based on rates
        prob = self.rates * (bm.dt / 1000.)  # Convert Hz to probability
        self.spike.value = bm.random.rand(self.num) < prob
        return self.spike.value

# Simple Synapse Model

class SimpleSynapse(bp.dyn.SynConn):
    """Simple exponential synapse"""
    def __init__(self, pre, post, conn, g_max=1., tau_syn=5.):
        super().__init__(pre=pre, post=post, conn=conn)
        
        self.g_max = g_max
        self.tau_syn = tau_syn
        
        # Get connection matrix
        self.conn_mat = self.conn.require('conn_mat')
        
        # Initialize weights
        self.weights = bm.Variable(
            self.conn_mat * bm.random.uniform(0.5, 1.0, self.conn_mat.shape) * g_max
        )
        
    def update(self):
        # Simple synaptic transmission
        # Note: No conditional needed - if no spikes, dot product is zero
        spike_input = bm.dot(self.pre.spike.astype(bm.float32), 
                           self.weights.astype(bm.float32))
        self.post.input_current += spike_input

# Simple Network Model

class SimpleSpikingNetwork(bp.Network):
    """Simple 2-layer spiking neural network"""
    def __init__(self, n_input=2, n_hidden=10, n_output=2, params=None):
        super().__init__()
        
        # Default parameters
        if params is None:
            params = {
                'input_rate_scale': 100.0,   # Max firing rate for inputs
                'tau_hidden': 20.0,          # Hidden neuron time constant
                'tau_output': 15.0,          # Output neuron time constant
                'g_input_hidden': 15.0,      # Input->Hidden synaptic strength
                'g_hidden_output': 20.0,     # Hidden->Output synaptic strength
                'g_lateral_inhibition': -5.0,# Lateral inhibition strength
                'noise_scale': 0.5,          # Noise level in neurons
                'connection_prob': 0.8,      # Connection probability
            }
        self.params = params
        
        # Create layers
        self.input_layer = SimpleInputNeuron(size=n_input)
        
        self.hidden_layer = SimpleLIFNeuron(
            size=n_hidden,
            tau=params['tau_hidden'],
            noise_scale=params['noise_scale']
        )
        
        self.output_layer = SimpleLIFNeuron(
            size=n_output,
            tau=params['tau_output'],
            noise_scale=params['noise_scale']
        )
        
        # Create connections
        # Input to Hidden: All-to-all
        conn_ih = bp.conn.All2All()
        self.syn_ih = SimpleSynapse(
            self.input_layer, self.hidden_layer, conn_ih,
            g_max=params['g_input_hidden']
        )
        
        # Hidden to Output: Random connections
        conn_ho = bp.conn.FixedProb(prob=params['connection_prob'])
        self.syn_ho = SimpleSynapse(
            self.hidden_layer, self.output_layer, conn_ho,
            g_max=params['g_hidden_output']
        )
        
        # Lateral inhibition in output layer
        conn_oo = bp.conn.All2All(include_self=False)
        self.syn_oo = SimpleSynapse(
            self.output_layer, self.output_layer, conn_oo,
            g_max=params['g_lateral_inhibition']
        )
        
    def update(self):
        # Update in order
        self.input_layer.update()
        self.syn_ih.update()
        self.hidden_layer.update()
        self.syn_ho.update()
        self.syn_oo.update()
        self.output_layer.update()
        return self.output_layer.spike.value

# Training and Evaluation Functions

def encode_input(x, y, max_rate=100.0):
    """Encode 2D coordinates as firing rates"""
    # Simple linear encoding: map [-3, 3] to [0, max_rate]
    # You can experiment with different encoding schemes
    rate_x = (x + 3) / 6 * max_rate
    rate_y = (y + 3) / 6 * max_rate
    return np.array([rate_x, rate_y])

def evaluate_network(net, X_test, y_test, T_simulation=100.0, dt=0.1):
    """Evaluate network accuracy"""
    correct = 0
    predictions = []
    
    for i in range(len(X_test)):
        x, y = X_test[i]
        label = int(y_test[i])
        
        # Encode input
        rates = encode_input(x, y, net.params['input_rate_scale'])
        net.input_layer.rates[:] = rates
        
        # Reset network state
        net.hidden_layer.V[:] = net.hidden_layer.V_rest
        net.output_layer.V[:] = net.output_layer.V_rest
        net.hidden_layer.spike[:] = False
        net.output_layer.spike[:] = False
        net.hidden_layer.refractory[:] = 0.
        net.output_layer.refractory[:] = 0.
        
        # Run simulation
        runner = bp.DSRunner(
            net,
            monitors={'output_spikes': net.output_layer.spike},
            dt=dt,
            progress_bar=False
        )
        runner.run(T_simulation)
        
        # Count spikes
        spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
        prediction = np.argmax(spike_counts)
        predictions.append(prediction)
        
        if prediction == label:
            correct += 1
    
    accuracy = correct / len(X_test)
    return accuracy, np.array(predictions)

def visualize_dataset_and_results(X_train, y_train, X_test, y_test, predictions=None):
    """Visualize the dataset and classification results"""
    fig, axes = plt.subplots(1, 2 if predictions is None else 3, figsize=(12 if predictions is None else 18, 5))
    
    # Training data
    ax = axes[0]
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', 
                        alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_title('Training Data', fontsize=12)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax)
    
    # Test data
    ax = axes[1]
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', 
                        alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_title('Test Data (True Labels)', fontsize=12)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax)
    
    # Predictions
    if predictions is not None:
        ax = axes[2]
        # Show correct and incorrect predictions
        correct_mask = predictions == y_test
        ax.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1], 
                  c=predictions[correct_mask], cmap='viridis', alpha=0.6, 
                  marker='o', s=50, edgecolors='green', linewidth=2, label='Correct')
        ax.scatter(X_test[~correct_mask, 0], X_test[~correct_mask, 1], 
                  c=predictions[~correct_mask], cmap='viridis', alpha=0.6, 
                  marker='X', s=100, edgecolors='red', linewidth=2, label='Incorrect')
        ax.set_title(f'Predictions (Accuracy: {np.mean(correct_mask):.1%})', fontsize=12)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_network_activity(net, sample_point, T_simulation=200.0, dt=0.1):
    """Visualize network activity for a single input"""
    x, y = sample_point
    rates = encode_input(x, y, net.params['input_rate_scale'])
    net.input_layer.rates[:] = rates
    
    # Reset network
    net.hidden_layer.V[:] = net.hidden_layer.V_rest
    net.output_layer.V[:] = net.output_layer.V_rest
    
    # Run with monitoring
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
    runner.run(T_simulation)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 2, 2, 2], hspace=0.3)
    
    # Input visualization
    ax = fig.add_subplot(gs[0, :])
    ax.bar([0, 1], rates, color=['blue', 'orange'])
    ax.set_ylim(0, net.params['input_rate_scale'] * 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['X', 'Y'])
    ax.set_ylabel('Input Rate (Hz)')
    ax.set_title(f'Input Point: ({x:.2f}, {y:.2f})', fontsize=12)
    
    # Input spikes
    ax = fig.add_subplot(gs[1, :])
    times = runner.mon.ts
    for i in range(2):
        spike_times = times[runner.mon['input_spikes'][:, i]]
        ax.scatter(spike_times, [i] * len(spike_times), c='black', s=10, alpha=0.8)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['X', 'Y'])
    ax.set_ylabel('Input')
    ax.set_title('Input Spikes')
    ax.set_xlim(0, T_simulation)
    
    # Hidden layer activity
    ax = fig.add_subplot(gs[2, 0])
    spike_times, spike_neurons = np.where(runner.mon['hidden_spikes'])
    if len(spike_times) > 0:
        ax.scatter(times[spike_times], spike_neurons, c='blue', s=5, alpha=0.6)
    ax.set_ylabel('Hidden Neuron')
    ax.set_title('Hidden Layer Spikes')
    ax.set_xlim(0, T_simulation)
    
    # Hidden layer voltages (sample neurons)
    ax = fig.add_subplot(gs[2, 1])
    n_sample = min(5, net.hidden_layer.num)
    for i in range(n_sample):
        ax.plot(times, runner.mon['hidden_V'][:, i], alpha=0.7, linewidth=1)
    ax.axhline(y=net.hidden_layer.V_th, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Hidden Layer Voltages (sample)')
    ax.set_xlim(0, T_simulation)
    ax.legend()
    
    # Output layer
    ax = fig.add_subplot(gs[3, 0])
    colors = ['green', 'red']
    for i in range(net.output_layer.num):
        spike_times = times[runner.mon['output_spikes'][:, i]]
        ax.scatter(spike_times, [i] * len(spike_times), c=colors[i], s=20, alpha=0.8, 
                  label=f'Class {i}')
    ax.set_ylim(-0.5, net.output_layer.num - 0.5)
    ax.set_ylabel('Output Neuron')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Output Layer Spikes')
    ax.set_xlim(0, T_simulation)
    ax.legend()
    
    # Output voltages and spike counts
    ax = fig.add_subplot(gs[3, 1])
    spike_counts = np.sum(runner.mon['output_spikes'], axis=0)
    bars = ax.bar(range(net.output_layer.num), spike_counts, color=colors[:net.output_layer.num])
    ax.set_xlabel('Output Neuron (Class)')
    ax.set_ylabel('Total Spike Count')
    ax.set_title(f'Output Spike Counts (Winner: Class {np.argmax(spike_counts)})')
    ax.set_xticks(range(net.output_layer.num))
    
    # Add text with counts
    for i, (bar, count) in enumerate(zip(bars, spike_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(int(count)), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return np.argmax(spike_counts)

def parameter_sensitivity_analysis(X_train, y_train, X_test, y_test, base_params):
    """Analyze how different parameters affect network performance"""
    parameters_to_test = {
        'input_rate_scale': [50, 100, 150, 200],
        'g_input_hidden': [5, 10, 15, 20, 25],
        'g_hidden_output': [10, 15, 20, 25, 30],
        'g_lateral_inhibition': [-10, -5, -2, -1, 0],
        'tau_hidden': [10, 20, 30, 40],
        'noise_scale': [0.1, 0.5, 1.0, 2.0],
    }
    
    results = {}
    
    for param_name, param_values in parameters_to_test.items():
        print(f"\nTesting parameter: {param_name}")
        accuracies = []
        
        for value in param_values:
            # Copy base parameters and modify the one we're testing
            test_params = base_params.copy()
            test_params[param_name] = value
            
            # Create and evaluate network
            net = SimpleSpikingNetwork(n_hidden=15, params=test_params)
            accuracy, _ = evaluate_network(net, X_test, y_test, T_simulation=100.0)
            accuracies.append(accuracy)
            print(f"  {param_name} = {value}: {accuracy:.1%}")
        
        results[param_name] = (param_values, accuracies)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (param_name, (values, accuracies)) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(values, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Effect of {param_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Mark the base parameter value
        if param_name in base_params:
            ax.axvline(x=base_params[param_name], color='red', linestyle='--', 
                      alpha=0.5, label='Base value')
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

# Save/Load Functions

def save_network(net, accuracy, dataset_type, filename=None):
    """Save network parameters and weights"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_snn_{dataset_type}_{timestamp}.pkl"
    
    save_data = {
        'params': net.params,
        'accuracy': accuracy,
        'dataset_type': dataset_type,
        'weights': {
            'input_hidden': net.syn_ih.weights.value.copy(),
            'hidden_output': net.syn_ho.weights.value.copy(),
            'output_lateral': net.syn_oo.weights.value.copy(),
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Network saved to: {filename}")
    return filename

def load_network(filename):
    """Load network from file"""
    with open(filename, 'rb') as f:
        save_data = pickle.load(f)
    
    net = SimpleSpikingNetwork(n_hidden=15, params=save_data['params'])
    
    # Restore weights
    net.syn_ih.weights.value = save_data['weights']['input_hidden']
    net.syn_ho.weights.value = save_data['weights']['hidden_output']
    net.syn_oo.weights.value = save_data['weights']['output_lateral']
    
    return net, save_data['accuracy'], save_data['dataset_type']

# Main Application

def main():
    print("="*60)
    print("Simple Spiking Neural Network Demo")
    print("="*60)
    
    # Choose dataset
    print("\nChoose a dataset:")
    print("1. Circles (easiest)")
    print("2. Moons")
    print("3. Blobs")
    print("4. Spiral (hardest)")
    
    dataset_choice = input("\nEnter choice (1-4): ") or "1"
    dataset_types = ['circles', 'moons', 'blobs', 'spiral']
    dataset_type = dataset_types[int(dataset_choice) - 1]
    
    # Generate dataset
    print(f"\nGenerating {dataset_type} dataset...")
    X_train, X_test, y_train, y_test = generate_simple_dataset(dataset_type, n_samples=200)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Visualize dataset
    visualize_dataset_and_results(X_train, y_train, X_test, y_test)
    
    # Define base parameters
    base_params = {
        'input_rate_scale': 100.0,
        'tau_hidden': 20.0,
        'tau_output': 15.0,
        'g_input_hidden': 15.0,
        'g_hidden_output': 20.0,
        'g_lateral_inhibition': -5.0,
        'noise_scale': 0.5,
        'connection_prob': 0.8,
    }
    
    # Main menu
    while True:
        print("\n" + "="*40)
        print("What would you like to do?")
        print("1. Run with default parameters")
        print("2. Tune individual parameters")
        print("3. Run parameter sensitivity analysis")
        print("4. Visualize single point classification")
        print("5. Save current network")
        print("6. Load saved network")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ")
        
        if choice == '1':
            # Run with default parameters
            print("\nCreating network with default parameters...")
            net = SimpleSpikingNetwork(n_hidden=15, params=base_params)
            
            print("Evaluating network...")
            accuracy, predictions = evaluate_network(net, X_test, y_test)
            print(f"\nAccuracy: {accuracy:.1%}")
            
            # Visualize results
            visualize_dataset_and_results(X_train, y_train, X_test, y_test, predictions)
            
        elif choice == '2':
            # Interactive parameter tuning
            print("\nCurrent parameters:")
            for key, value in base_params.items():
                print(f"  {key}: {value}")
            
            print("\nWhich parameter to modify?")
            param_name = input("Parameter name (or 'done' to finish): ").strip()
            
            if param_name in base_params:
                try:
                    new_value = float(input(f"New value for {param_name}: "))
                    base_params[param_name] = new_value
                    
                    # Test with new parameters
                    net = SimpleSpikingNetwork(n_hidden=15, params=base_params)
                    accuracy, predictions = evaluate_network(net, X_test, y_test)
                    print(f"\nAccuracy with {param_name}={new_value}: {accuracy:.1%}")
                    
                    visualize_dataset_and_results(X_train, y_train, X_test, y_test, predictions)
                except ValueError:
                    print("Invalid value!")
            
        elif choice == '3':
            # Parameter sensitivity analysis
            print("\nRunning parameter sensitivity analysis...")
            print("This will test how each parameter affects accuracy...")
            parameter_sensitivity_analysis(X_train, y_train, X_test, y_test, base_params)
            
        elif choice == '4':
            # Visualize single point
            print("\nEnter coordinates to classify:")
            try:
                x = float(input("X coordinate (-3 to 3): "))
                y = float(input("Y coordinate (-3 to 3): "))
                
                net = SimpleSpikingNetwork(n_hidden=15, params=base_params)
                prediction = visualize_network_activity(net, (x, y))
                
                print(f"\nPredicted class: {prediction}")
                
            except ValueError:
                print("Invalid coordinates!")
                
        elif choice == '5':
            # Save network
            net = SimpleSpikingNetwork(n_hidden=15, params=base_params)
            accuracy, _ = evaluate_network(net, X_test, y_test)
            save_network(net, accuracy, dataset_type)
            
        elif choice == '6':
            # Load network
            saved_files = [f for f in os.listdir('.') if f.startswith('simple_snn_') and f.endswith('.pkl')]
            if not saved_files:
                print("No saved networks found!")
            else:
                print("\nSaved networks:")
                for i, f in enumerate(saved_files):
                    print(f"{i+1}. {f}")
                
                try:
                    idx = int(input("Choose file number: ")) - 1
                    net, saved_accuracy, saved_dataset = load_network(saved_files[idx])
                    print(f"Loaded network (accuracy: {saved_accuracy:.1%}, dataset: {saved_dataset})")
                    
                    # Update base_params
                    base_params = net.params.copy()
                    
                except (ValueError, IndexError):
                    print("Invalid choice!")
                    
        elif choice == '7':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
