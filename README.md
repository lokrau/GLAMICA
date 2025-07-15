# Gaze and Language in Action: Multimodal Interfaces for Cognitive (GLAMICA) 
## By: Lorenz Krause, Waqar Quershi, and Michael Rice
## Aria Client SDK (given by Meta Project Aria Team)

The Aria Client SDK with CLI provides robust capabilities for creating computer vision
and machine learning applications with Project Aria glasses.

## Usage:

1. Install the Project Aria Client SDK by following the instructions in the [official documentation](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup).

2. In the project directory, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. In the project directory, run the following command to start the stream from the glasses:
    ```bash
    aria streaming start --interface usb --use-ephemeral-certs 
    ```

4. When this is finished, you can connect to the stream and start the GLAMICA app by running:
    ```bash
    python -m glamica
    ```

Now a web interface will be opened in your browser where you can interact with the system.