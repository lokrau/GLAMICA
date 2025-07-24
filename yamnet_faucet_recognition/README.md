# Fine-tuning the model for faucet sound recognition
## Author: Lorenz Krause

This folder contains a total of 20 minutes and 8 seconds of audio data in the `data` folder, which is used to fine-tune a model for faucet sound recognition based on the YAMNet architecture for feature extraction followed by a simple fully connected classifier. Of this 20 minutes and 8 seconds, 10 minutes and 4 seconds are examples for faucet sounds, and 10 minutes and 4 seconds are examples for non-faucet sounds. For the training process do the following steps:

1. Run the `resample_data.ipynb` notebook which resamples the audio data to 16 kHz, splits all the data into 2-second blocks, and saves it in the `data_split` folder.

2. Run the `training.ipynb` notebook which splits the data into 80% training, 10% validation, and 10% test sets, trains the model, and saves the trained model in the `model` folder.