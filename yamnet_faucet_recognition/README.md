# Fine-tuning the model for faucet sound recognition
## Author: Lorenz Krause

This folder contains a total of 20 minutes and 8 seconds of audio data in the `data` folder, which is used to fine-tune a model for faucet sound recognition based on the YAMNet architecture for feature extraction followed by a simple fully connected classifier. Of this 20 minutes and 8 seconds, 10 minutes and 4 seconds are examples for faucet sounds, and 10 minutes and 4 seconds are examples for non-faucet sounds. To record data using the Project Aria glasses `stream_sound_2_sec_blocks.py` can be used, which records 2-second audio blocks and saves them in the a folder that can be specified in the code. To use this code, do the following steps:
1. Specify the folder where the audio should be saved.

2. In terminal run the following command to start the stream from the glasses:
    1. If you connect to the glasses via USB:
    ```bash
        aria streaming start --interface usb --use-ephemeral-certs --profile profile18
    ```
    2. If you connect to the glasses via WiFi (DEVICE_IP can be found in the Project Aria app):
    ```bash
        aria streaming start --interface wifi --use-ephemeral-certs --device-ip [DEVICE_IP] --profile profile18
    ```

3. After stream started run the following command to start the recording:
```bash
    python stream_sound_2_sec_blocks.py
```


For the training process do the following steps:

1. Run the `resample_data.ipynb` notebook which resamples the audio data to 16 kHz, splits all the data into 2-second blocks, and saves it in the `data_split` folder.

2. Run the `training.ipynb` notebook which splits the data into 80% training, 10% validation, and 10% test sets, trains the model, and saves the trained model in the `model` folder.