# Build Your Own Brain - Spiking Neural Network for Digit Classification

A biologically-inspired spiking neural network implementation using BrainPy for handwritten digit recognition on the scikit-learn digits dataset.

## Overview

This project implements a multi-layer spiking neural network with:
- **Leaky Integrate-and-Fire (LIF) neurons** with adaptive thresholds and heterogeneous parameters
- **Spike-Timing Dependent Plasticity (STDP)** for unsupervised learning
- **Poisson input encoding** for converting pixel intensities to spike trains
- **Lateral inhibition** for competition between neurons
- **Real-time visualization** of network activity and learning

## Features

### Neural Dynamics
- **LIF Neurons**: Biologically plausible neuron model with:
  - Membrane potential dynamics
  - Refractory periods
  - Spike-frequency adaptation
  - Heterogeneous parameters to ensure diverse firing patterns

### Network Architecture
- **Input Layer**: 64 Poisson neurons (8×8 pixel input)
- **Hidden Layer**: 100 LIF neurons with lateral inhibition
- **Output Layer**: 10 LIF neurons (one per digit class)
- **Synaptic Connections**: 
  - Exponential synapses with configurable delays
  - Optional STDP learning for input→hidden connections
  - Winner-take-all dynamics in output layer

### Visualization Tools
- Spike raster plots for all layers
- Membrane potential traces
- Weight distribution analysis
- Population activity dynamics
- Classification performance metrics

## Requirements

```bash
pip install brainpy
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scipy
```

## Usage

### Basic Classification Demo
```python
python main.py
```

This will:
1. Load and preprocess the digits dataset
2. Create a spiking neural network
3. Run inference on test samples
4. Display classification results and network activity
5. Optionally test STDP learning

### Key Parameters

```python
# Network configuration
net = SpikingDigitClassifier(
    n_input=64,        # Input neurons (8×8 pixels)
    n_hidden=100,      # Hidden layer size
    n_output=10,       # Output classes
    use_stdp=True,     # Enable spike-timing dependent plasticity
    connection_prob=0.5 # Sparse connectivity
)

# Neuron parameters
LIFNeuron(
    V_rest=-65.,       # Resting potential (mV)
    V_th=-50.,         # Spike threshold (mV)
    tau=20.,           # Membrane time constant (ms)
    heterogeneity=0.1  # Parameter variation (0-1)
)
```

## How It Works

1. **Input Encoding**: Pixel intensities are converted to Poisson spike trains with rates proportional to intensity
2. **Signal Propagation**: Spikes propagate through the network layers via synaptic connections
3. **Competition**: Lateral inhibition creates competition between neurons in the same layer
4. **Classification**: The output neuron with the highest spike count determines the predicted digit
5. **Learning** (optional): STDP adjusts synaptic weights based on spike timing correlations

## Example Output

```
Running inference on 20 test samples...
Sample 1: True=7, Predicted=7, Correct=✓
Sample 2: True=4, Predicted=4, Correct=✓
Sample 3: True=1, Predicted=8, Correct=✗
...
Accuracy: 65.00%
```

## Biological Inspiration

This implementation incorporates several biologically-inspired mechanisms:
- **Heterogeneous neurons**: Like real neurons, each unit has slightly different parameters
- **Spike-based computation**: Information is encoded in spike timing and rates
- **Local learning rules**: STDP mimics synaptic plasticity in the brain
- **Lateral inhibition**: Creates competition similar to cortical circuits

## Troubleshooting

If all output neurons show identical firing:
- Increase `heterogeneity` parameter in neuron initialization
- Check that lateral inhibition weights are properly configured
- Ensure input spike rates are sufficiently high (>50 Hz average)
- Verify that synaptic weights have appropriate initial values

## Future Enhancements

- Convolutional spike layers for better feature extraction
- Homeostatic plasticity for stable learning
- GPU acceleration via JAX backend
- Support for larger image datasets (MNIST, CIFAR)
- Online learning during inference

## License

MIT © [jwest33](https://github.com/jwest33)

## Acknowledgments

Built with [BrainPy](https://github.com/brainpy/BrainPy)
