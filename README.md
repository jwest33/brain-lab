# Build Your Own Brain - Spiking Neural Network for Digit Classification

A biologically-inspired spiking neural network implementation using BrainPy for handwritten digit recognition on the scikit-learn digits dataset, featuring model persistence and continuous learning capabilities.

## Overview

This project implements a multi-layer spiking neural network with:
- **Leaky Integrate-and-Fire (LIF) neurons** with adaptive thresholds and heterogeneous parameters
- **Spike-Timing Dependent Plasticity (STDP)** for unsupervised learning
- **Model persistence** for saving and loading trained networks
- **Poisson input encoding** for converting pixel intensities to spike trains
- **Convolution-style connectivity** from input to hidden layer
- **Lateral inhibition** for competition between neurons
- **Real-time visualization** of network activity and learning

## Features

### Neural Dynamics
- **LIF Neurons**: Biologically plausible neuron model with:
  - Membrane potential dynamics with proper time-based integration
  - Refractory periods measured in milliseconds
  - Spike-frequency adaptation for realistic firing patterns
  - Heterogeneous parameters to ensure diverse neural responses
  - Intrinsic excitability variations

### Network Architecture
- **Input Layer**: 64 Poisson neurons (8×8 pixel input)
- **Hidden Layer**: 36 LIF neurons (6×6 spatial arrangement) with lateral inhibition
- **Output Layer**: 10 LIF neurons (one per digit class)
- **Synaptic Connections**: 
  - Convolution-style input→hidden connectivity (3×3 kernels with stride 1)
  - Exponential synapses with configurable delays
  - Optional STDP learning for input→hidden connections
  - Sparse hidden→output connectivity (70% connection probability)
  - Winner-take-all dynamics in output layer with distance-dependent inhibition

### Model Management
- **Save/Load Functionality**: Complete network state persistence including:
  - All synaptic weights across layers
  - Neuron heterogeneity parameters
  - Connection matrices
  - Biases and adaptation parameters
- **Model History**: Timestamped model files with creation dates
- **Continuous Learning**: Load pre-trained models and continue training
- **Model Selection Interface**: Interactive menu for managing saved models
- **Automatic Cleanup**: Option to remove old models when exceeding 5 saved files

### Visualization Tools
- Spike raster plots for all layers
- Membrane potential traces with spike markers
- Weight distribution analysis
- Population activity dynamics
- STDP weight change visualization (before/after and delta)
- Classification confusion matrix
- Per-class firing rate analysis

## Requirements

```bash
pip install brainpy
pip install numpy
pip install matplotlib
pip install scikit-learn
```

## Usage

### Running the Application
```python
python main.py
```

The main menu offers three options:
1. **Run classification demo** - Test the network on digit samples
2. **Train/Load STDP model** - Train new or continue training existing models
3. **Analyze existing model** - Load and analyze saved models

### Model Management

#### Training a New Model
- Select option 2 from main menu
- Choose "0" when prompted for model selection
- Configure training parameters (epochs, samples)
- Model will be saved automatically after training

#### Loading and Continuing Training
- Select option 2 from main menu
- Choose from list of saved models
- Decide whether to continue training or just evaluate
- Updated model can be saved with new timestamp

#### Saved Model Format
Models are saved as `stdp_model_YYYYMMDD_HHMMSS.pkl` containing:
- Network architecture parameters
- All synaptic weights
- Neuron parameters and heterogeneity
- Connection matrices
- Training state

### Key Parameters

```python
# Network configuration
net = SpikingDigitClassifier(
    n_input=64,         # Input neurons (8×8 pixels)
    n_output=10,        # Output classes
    use_stdp=True       # Enable spike-timing dependent plasticity
)
# Note: n_hidden is automatically calculated as 36 (6×6) based on convolution parameters

# Neuron parameters
LIFNeuron(
    V_rest=-65.,        # Resting potential (mV)
    V_th=-50.,          # Spike threshold (mV)
    tau=20.,            # Membrane time constant (ms)
    tau_ref=5.,         # Refractory period (ms)
    tau_adapt=100.,     # Adaptation time constant (ms)
    heterogeneity=0.1   # Parameter variation (0-1)
)

# Convolution parameters (input to hidden)
kernel_size = 3         # 3×3 receptive fields
stride = 1              # No overlap between kernels

# Training parameters
n_epochs = 5            # Training epochs
n_train_samples = 100   # Samples per epoch
T_present = 200.0       # Presentation time per sample (ms)
dt = 0.1                # Simulation time step (ms)
```

## How It Works

1. **Input Encoding**: Pixel intensities are converted to Poisson spike trains with rates proportional to intensity (0-150 Hz)

2. **Convolution-Style Processing**: Each hidden neuron receives input from a 3×3 patch of input neurons, creating local receptive fields

3. **Signal Propagation**: Spikes propagate through exponential synapses with proper temporal dynamics

4. **Neural Integration**: LIF neurons integrate inputs according to differential equations with time-based decay

5. **Competition**: 
   - Hidden layer: Weak lateral inhibition for sparse coding
   - Output layer: Strong winner-take-all dynamics with distance-dependent inhibition

6. **Classification**: Output neuron with highest spike count determines predicted digit (with random tie-breaking for equal counts)

7. **Learning** (STDP): Synaptic weights are modified based on precise spike timing:
   - Pre-before-post: Long-term potentiation (LTP)
   - Post-before-pre: Long-term depression (LTD)

8. **State Persistence**: Complete network state can be saved and restored for continued use

## Network Reset Protocol

Between samples, the network undergoes complete state reset:
- Membrane potentials reset to resting values with small random variations
- Spike states and refractory periods cleared
- Adaptation variables reset with small random initialization
- Synaptic conductances reset to zero
- Input rates updated for new sample

## Custom Connection Implementation

The network uses a custom `MatrixConn` class that inherits from BrainPy's `TwoEndConnector` to implement the convolution-style connectivity pattern. This allows precise control over which input neurons connect to which hidden neurons while maintaining compatibility with BrainPy's connection system.

## Example Output

```
Extended LIF Network with Digit Classification
==================================================

What would you like to do?
1. Run classification demo
2. Train/Load STDP model
3. Analyze existing model

Enter your choice (1-3): 1

Loading digits dataset...
Dataset shape: (1437, 64)
Number of classes: 10

Creating spiking neural network...

Running inference on 20 test samples...
Sample 1: True=3, Predicted=3, Correct=✓
  Debug - Output spike counts: [12  8 14 25  9 15 11 13 10 18]
  Debug - Max output voltage: -47.82
  Debug - Input spike rate: 7.8 spikes
  Debug - Hidden spike rate: 15.2 spikes
  Debug - Total input spikes: 4982
Sample 2: True=0, Predicted=0, Correct=✓
...

Accuracy: 30.00%

Analyzing network activity patterns...
[Visualizations appear]
```

## Biological Inspiration

This implementation incorporates several biologically-inspired mechanisms:
- **Local receptive fields**: Convolution-style connectivity mimics how neurons in visual cortex respond to local image patches
- **Heterogeneous neurons**: Each neuron has unique parameters mimicking biological diversity
- **Temporal dynamics**: All processes respect biological time scales (milliseconds)
- **Spike-based computation**: Information encoded in precise spike timing and rates
- **Local learning rules**: STDP implements Hebbian learning ("neurons that fire together, wire together")
- **Lateral inhibition**: Creates competition similar to cortical circuits
- **Adaptation**: Neurons adjust their excitability based on recent activity

## Troubleshooting

### Low or Random Accuracy
- The base network without training typically achieves 20-40% accuracy
- Increase `heterogeneity` parameter to ensure diverse neural responses
- Verify lateral inhibition strength (adjust `g_max` for inhibitory connections)
- Check input encoding rates (should see ~5-10 spikes per active pixel)
- Ensure proper network reset between samples

### No Learning Progress with STDP
- Increase training epochs or samples per epoch
- Adjust STDP parameters (`A_pre`, `A_post`)
- Verify spike timing windows (`tau_pre`, `tau_post`)
- Check that input patterns generate sufficient spiking activity
- Monitor weight changes during training

### Model Loading Issues
- Ensure pickle files are not corrupted
- Verify BrainPy version compatibility
- Check that model architecture matches saved parameters
- Note that the network automatically handles connection matrix restoration

### Connection Errors
- The custom `MatrixConn` class handles BrainPy's connection interface
- Ensures compatibility with both tuple and integer size specifications
- Connection matrices are properly saved and restored with models

## Network Analysis Features

The analysis tools provide insights into:
- **Firing rate matrix**: Shows how each output neuron responds to each digit class
- **Weight distributions**: Visualizes synaptic strength patterns
- **Population dynamics**: Temporal evolution of network activity
- **Spike count distributions**: Statistical analysis of neural activity levels

## Future Enhancements

- Implement more sophisticated convolution patterns (different kernel sizes, strides)
- Add pooling layers for hierarchical feature extraction
- Homeostatic plasticity for stable long-term learning
- GPU acceleration via JAX backend
- Support for larger image datasets (MNIST, Fashion-MNIST)
- Online learning during inference
- Ensemble methods using multiple saved models
- Visualization of learned convolution kernels
- Support for different neuron models (Izhikevich, AdEx)
- Implement backpropagation-through-time for supervised learning

## License

MIT © [jwest33](https://github.com/jwest33)

## Acknowledgments

Built with [BrainPy](https://github.com/brainpy/BrainPy) - Brain Dynamics Programming in Python
