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

## MIDI Processing and Analysis

### MIDI to Notes Conversion
The `midi_to_notes` function extracts relevant information from MIDI files, including pitch, start time, end time, step, and duration.

### Piano Roll Visualization
The `plot_piano_roll` function generates a piano roll representation of the notes extracted from MIDI files.

## LSTM Model Training

### Sequence Generation
The project utilizes TensorFlow's `tf.data.Dataset` to create sequences of notes for training the LSTM model.

### Custom Loss Function
A custom loss function, `mse_with_positive_pressure`, is defined to incorporate positive pressure on the predicted values.

### Model Architecture
The LSTM model is designed with three output layers corresponding to pitch, step, and duration predictions.

### Model Training
The model is trained with a combination of sparse categorical cross-entropy and mean squared error loss functions. <br>
<b>Note: </b> <i>I have interrupted the training at ```epoch 21/50``` to see how it would affect the generator. This also resulted in a ```Keyboard Interrupt``` error. This can be avoided by letting the training complete to 50 epochs.</i>. 

## Training Results

### Loss Evaluation
The trained model's performance is evaluated on the training dataset, and the loss values for pitch, step, duration, and the total loss are presented.

### Loss Weight Adjustment
The loss weights are fine-tuned to balance the contributions of pitch, step, and duration to the overall loss.

### Training History
The training history, including total loss over epochs, is visualized using matplotlib.

## Music Generation

### Next Note Prediction
The trained model is used to predict the next musical note in a sequence, considering temperature as a parameter for randomness.

### Generated Composition
A specified number of notes are predicted, and the resulting musical composition is converted back to a MIDI file.

### Audio Playback
The generated MIDI file is played back using the `fluidsynth` library for audio synthesis.

## Conclusion

This project provides a comprehensive overview of the process involved in training an LSTM neural network for music generation. It covers data preparation, model architecture, training, and the generation of new musical compositions. The code is well-documented and structured for ease of understanding and further exploration.

# Setup

### Dependencies
- [pretty_midi](https://craffel.github.io/pretty-midi/)
- [FluidSynth](https://www.fluidsynth.org/api/modules.html)

### Installation
```bash
sudo apt install -y fluidsynth
pip install --upgrade pyfluidsynth
pip install pretty_midi
```

## Usage

1. **Download MAESTRO Dataset:** If the MAESTRO dataset is not found, the script will automatically download it. The dataset is used for training the model.

2. **Choose a MIDI File:** The script randomly selects a MIDI file from the dataset for processing. You can change the file number in the range [0, len(filenames)].

3. **Display Original Audio:** Use the `display_audio` function to listen to a 30-second audio snippet of the selected MIDI file.

4. **Extract Notes from MIDI:** The `midi_to_notes` function extracts note information (pitch, start time, end time, step, duration) from the MIDI file.

5. **Visualize Piano Roll:** The `plot_piano_roll` function generates a piano roll visualization of the extracted notes.

6. **Train the LSTM Model:** The script then proceeds to train an LSTM model using TensorFlow. The model is trained to predict the next musical note in the sequence.

7. **Generate New Sequence:** After training, the model is used to generate a new sequence of MIDI notes. The `predict_next_note` function is employed for this purpose.

8. **Display Generated Audio:** The generated sequence is converted back to a MIDI file, and the resulting audio is played using the `display_audio` function.

## Results
The generated MIDI sequence showcases the model's ability to learn musical patterns and create new compositions inspired by the input dataset.

Feel free to experiment with different hyperparameters, model architectures, and training durations to further enhance the quality of generated sequences.


