# Build Your Own Brain - Spiking Neural Network for Digit Classification

A biologically-inspired spiking neural network implementation using BrainPy for handwritten digit recognition on the scikit-learn digits dataset, featuring model persistence and continuous learning capabilities.

## Overview

This project implements a multi-layer spiking neural network with:
- **Leaky Integrate-and-Fire (LIF) neurons** with adaptive thresholds and heterogeneous parameters
- **Spike-Timing Dependent Plasticity (STDP)** for unsupervised learning
- **Model persistence** for saving and loading trained networks
- **Poisson input encoding** for converting pixel intensities to spike trains
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
- **Hidden Layer**: 100 LIF neurons with lateral inhibition
- **Output Layer**: 10 LIF neurons (one per digit class)
- **Synaptic Connections**: 
  - Exponential synapses with configurable delays
  - Optional STDP learning for input->hidden connections
  - Winner-take-all dynamics in output layer with Mexican-hat inhibition profile
  - Structured receptive fields (center-surround, edge detectors)

### Model Management
- **Save/Load Functionality**: Complete network state persistence including:
  - All synaptic weights across layers
  - Neuron heterogeneity parameters
  - Connection matrices
  - Biases and adaptation parameters
- **Model History**: Timestamped model files with creation dates
- **Continuous Learning**: Load pre-trained models and continue training
- **Model Selection Interface**: Interactive menu for managing saved models

### Visualization Tools
- Spike raster plots for all layers
- Membrane potential traces with spike markers
- Weight distribution analysis
- Population activity dynamics
- STDP weight change visualization
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
    n_hidden=100,       # Hidden layer size
    n_output=10,        # Output classes
    use_stdp=True,      # Enable spike-timing dependent plasticity
    connection_prob=0.5 # Sparse connectivity
)

# Neuron parameters
LIFNeuron(
    V_rest=-65.,        # Resting potential (mV)
    V_th=-50.,          # Spike threshold (mV)
    tau=20.,            # Membrane time constant (ms)
    tau_ref=5.,         # Refractory period (ms)
    tau_adapt=100.,     # Adaptation time constant (ms)
    heterogeneity=0.1   # Parameter variation (0-1)
)

# Training parameters
n_epochs = 5            # Training epochs
n_train_samples = 100   # Samples per epoch
T_present = 200.0       # Presentation time per sample (ms)
```

## How It Works

1. **Input Encoding**: Pixel intensities are converted to Poisson spike trains with rates proportional to intensity (0-150 Hz)
2. **Signal Propagation**: Spikes propagate through exponential synapses with proper temporal dynamics
3. **Neural Integration**: LIF neurons integrate inputs according to differential equations with time-based decay
4. **Competition**: Lateral inhibition creates winner-take-all dynamics with distance-dependent inhibition strength
5. **Classification**: Output neuron with highest spike count determines predicted digit (with random tie-breaking)
6. **Learning** (STDP): Synaptic weights are modified based on precise spike timing:
   - Pre-before-post: Long-term potentiation (LTP)
   - Post-before-pre: Long-term depression (LTD)
7. **State Persistence**: Complete network state can be saved and restored for continued use

## Network Reset Protocol

Between samples, the network undergoes complete state reset:
- Membrane potentials reset to resting values with small random variations
- Spike states and refractory periods cleared
- Adaptation variables reset with small random initialization
- Synaptic conductances reset to zero
- Input rates updated for new sample

## Example Output

```
What would you like to do?
1. Run classification demo
2. Train/Load STDP model
3. Analyze existing model

Enter your choice (1-3): 2

Saved Models Found:
==================================================
1. stdp_model_20240118_143022.pkl (created: 2024-01-18 14:30:22)
2. stdp_model_20240118_152105.pkl (created: 2024-01-18 15:21:05)

0. Train a new model

Enter your choice (0 to train new, or model number to load): 1

Loading model: stdp_model_20240118_143022.pkl
Model loaded from: stdp_model_20240118_143022.pkl

Continue training this model? (y/n): y

Evaluating current performance...
Current accuracy: 14.00%

Training with STDP...
Number of training epochs (default 5): 5
Number of training samples per epoch (default 100): 100

Epoch 1/5
  Processed 20/100 samples, running accuracy: 0.00%
  Processed 40/100 samples, running accuracy: 7.50%
  Processed 60/100 samples, running accuracy: 6.67%
  Processed 80/100 samples, running accuracy: 8.75%
  Processed 100/100 samples, running accuracy: 10.00%
  Test accuracy after epoch 1: 12.00%

..

Final evaluation...
Accuracy: 14.00% → 20.00% (+6.00% improvement)

Save this model? (y/n): y
Model saved to: stdp_model_20240118_160532.pkl
```

## Biological Inspiration

This implementation incorporates several biologically-inspired mechanisms:
- **Heterogeneous neurons**: Each neuron has unique parameters mimicking biological diversity
- **Temporal dynamics**: All processes respect biological time scales (milliseconds)
- **Spike-based computation**: Information encoded in precise spike timing and rates
- **Local learning rules**: STDP implements Hebbian learning ("neurons that fire together, wire together")
- **Lateral inhibition**: Creates competition similar to cortical circuits
- **Adaptation**: Neurons adjust their excitability based on recent activity
- **Synaptic delays**: Signal transmission takes time, as in real neural circuits

## Troubleshooting

### Low or Uniform Accuracy
- Increase `heterogeneity` parameter to ensure diverse neural responses
- Verify lateral inhibition strength (increase `g_max` for output layer inhibition)
- Check input encoding rates (should average 50-100 Hz for active pixels)
- Ensure proper network reset between samples

### No Learning Progress with STDP
- Increase training epochs or samples per epoch
- Adjust STDP parameters (`A_pre`, `A_post`)
- Verify spike timing windows (`tau_pre`, `tau_post`)
- Check that input patterns generate sufficient spiking activity

### Model Loading Issues
- Ensure pickle files are not corrupted
- Verify BrainPy version compatibility
- Check that model architecture matches saved parameters

## Future Enhancements

- Convolutional spike layers for better feature extraction
- Homeostatic plasticity for stable long-term learning
- GPU acceleration via JAX backend
- Support for larger image datasets (MNIST, CIFAR)
- Online learning during inference
- Ensemble methods using multiple saved models
- Visualization of learned receptive fields
- Support for different neuron models (Izhikevich, AdEx)

## License

MIT © [jwest33](https://github.com/jwest33)

## Acknowledgments

Built with [BrainPy](https://github.com/brainpy/BrainPy) - Brain Dynamics Programming in Python
