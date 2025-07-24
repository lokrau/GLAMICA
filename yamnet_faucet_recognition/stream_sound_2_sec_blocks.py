import argparse
import sys
import threading
import time
import os

import aria.sdk as aria
import numpy as np
import soundfile as sf

from common import quit_keypress, update_iptables

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    return parser.parse_args()

def max_signed_value_for_bytes(n):
    return (1 << (8 * n - 1)) - 1

def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    streaming_client = aria.StreamingClient()

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Audio
    config.message_queue_size[aria.StreamingDataType.Audio] = 10

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    class StreamingClientObserver:
        def __init__(self):
            self.audio_buffer = []
            self.audio_lock = threading.Lock()
            self.audio_max_value_ = max_signed_value_for_bytes(4)
            self.file_count = 0
            self.sample_rate = 48000
            self.block_size = 2 * self.sample_rate  # 2 seconds
            self.channels = 7

            if not os.path.exists("data/noise"):
                os.makedirs("data/noise")

        def on_audio_received(self, audio_data, timestamp_ns):
            audio_np = np.array(audio_data.data).astype(np.float64) / self.audio_max_value_

            try:
                audio_np = audio_np.reshape((-1, self.channels))
            except ValueError as e:
                print(f"Error reshaping audio: {e}")
                return

            with self.audio_lock:
                self.audio_buffer.append(audio_np)
                current_audio = np.vstack(self.audio_buffer)

                # If we have enough for 2 seconds, save and reset
                if current_audio.shape[0] >= self.block_size:
                    to_save = current_audio[:self.block_size]
                    sf.write(f"data/noise/block_{self.file_count:04d}.wav", to_save, samplerate=self.sample_rate)
                    print(f"Saved block_{self.file_count:04d}.wav")

                    self.file_count += 1
                    # Keep remaining samples for next block
                    self.audio_buffer = [current_audio[self.block_size:]]

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to audio data")
    streaming_client.subscribe()

    try:
        while not quit_keypress():
            time.sleep(1)
    finally:
        print("Stop listening to data")
        streaming_client.unsubscribe()

        # save remaining buffer as final partial block
        with observer.audio_lock:
            if observer.audio_buffer:
                final_audio = np.vstack(observer.audio_buffer)
                if final_audio.size > 0:
                    sf.write(f"data/noise/block_{observer.file_count:04d}_partial.wav", final_audio, samplerate=observer.sample_rate)
                    print(f"Saved final partial block block_{observer.file_count:04d}_partial.wav")

if __name__ == "__main__":
    main()
