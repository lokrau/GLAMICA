# Finding the IMU Bias of the Aria Glasses
## Author: Lorenz Krause

This directory contains the code to fine the IMU bias of the Aria glasses. The IMU bias is the offset that the IMU readings have, which can lead to drift over time. This code uses a simple method to find the bias by averaging the readings over a period of time when the glasses are stationary.

## Usage
1. Put the Aria glasses on a flat surface and make sure they are stationary.

2. Connect to the Aria glasses:
    ```bash
    aria streaming start --interface usb --use-ephemeral-certs
    ```

3. Start the IMU data collection:
    ```bash
    python -m my_streaming_test_gpt   
    ```

4. After one hour the collection automatically stops and the collected data is saved in `imu_data_one_hour.csv`.

5. Run the Jupyter notebook `get_average_imu_error.ipynb` to find the IMU bias. The notebook will read the collected data and calculate the bias for each axis (x, y, z) of the IMU.