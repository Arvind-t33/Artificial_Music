# MIDI Sequence Generation with LSTM

## Overview

This project focuses on generating MIDI sequences using a Long Short-Term Memory (LSTM) neural network. The goal is to create a model capable of learning musical patterns from existing MIDI files and then generate new sequences based on that learned knowledge.

## Requirements

- **FluidSynth**: Used for audio synthesis from MIDI files.
- **PyFluidSynth**: A Python wrapper for FluidSynth.
- **PrettyMIDI**: Library for handling MIDI files in a user-friendly manner.
- **TensorFlow**: Deep learning framework for building and training the LSTM model.
- **Seaborn, NumPy, Pandas, Matplotlib**: Data visualization and manipulation tools.

## Data Collection

The [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) is employed, containing a diverse collection of classical piano performances. MIDI files are downloaded and processed to extract musical notes.

# MIDI Processing and Analysis

## MIDI to Notes Conversion
The `midi_to_notes` function extracts relevant information from MIDI files, including pitch, start time, end time, step, and duration.

## Piano Roll Visualization
The `plot_piano_roll` function generates a piano roll representation of the notes extracted from MIDI files.

# LSTM Model Training

## Sequence Generation
The project utilizes TensorFlow's `tf.data.Dataset` to create sequences of notes for training the LSTM model.

## Custom Loss Function
A custom loss function, `mse_with_positive_pressure`, is defined to incorporate positive pressure on the predicted values.

## Model Architecture
The LSTM model is designed with three output layers corresponding to pitch, step, and duration predictions.

## Model Training
The model is trained with a combination of sparse categorical cross-entropy and mean squared error loss functions. 
<b>Note: <b> I have interrupted the training at ```epoch 21/50``` to see how it would affect the generator. This also resulted in a ```

# Training Results

## Loss Evaluation
The trained model's performance is evaluated on the training dataset, and the loss values for pitch, step, duration, and the total loss are presented.

## Loss Weight Adjustment
The loss weights are fine-tuned to balance the contributions of pitch, step, and duration to the overall loss.

## Training History
The training history, including total loss over epochs, is visualized using matplotlib.

# Music Generation

## Next Note Prediction
The trained model is used to predict the next musical note in a sequence, considering temperature as a parameter for randomness.

## Generated Composition
A specified number of notes are predicted, and the resulting musical composition is converted back to a MIDI file.

## Audio Playback
The generated MIDI file is played back using the `fluidsynth` library for audio synthesis.

# Conclusion

This project provides a comprehensive overview of the process involved in training an LSTM neural network for music generation. It covers data preparation, model architecture, training, and the generation of new musical compositions. The code is well-documented and structured for ease of understanding and further exploration.


