# Gaze and Language in Action: Multimodal Interfaces for Cognitive (GLAMICA) 
## By: Lorenz Krause, Waqar Shahid Qureshi, and Michael Rice
## Aria Client SDK (given by Meta Project Aria Team)

The Aria Client SDK with CLI provides robust capabilities for creating computer vision
and machine learning applications with Project Aria glasses. 
A example video of the GLAMICA application in use can be found [here](https://youtu.be/BfiTB2UJyzU).

## Usage:

1. Install the Project Aria Client SDK by following the instructions in the [official documentation](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup).

2. Make a file called `.env` in the root directory of this project and add your OpenAI API key:
    ```plaintext
        OPENAI_API_KEY=[YOUR_API_KEY]
    ```

3. In the project directory, install the required Python packages:
    ```bash
        pip install -r requirements.txt
    ```

4. In the project directory, run one of the following commands to start the stream from the glasses:
    1. If you connect to the glasses via USB:
    ```bash
        aria streaming start --interface usb --use-ephemeral-certs --profile profile18
    ```

    2. If you connect to the glasses via WiFi (DEVICE_IP can be found in the Project Aria app):
    ```bash
        aria streaming start --interface wifi --use-ephemeral-certs --device-ip [DEVICE_IP] --profile profile18
    ```

5. When this is finished, you can connect to the stream and start the GLAMICA app by running:
    ```bash
    python -m glamica
    ```

Now a web interface will be opened in your browser where you can interact with the system.