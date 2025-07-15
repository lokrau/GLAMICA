# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sys
import time
import csv

import aria.sdk as aria
from common import update_iptables
from projectaria_tools.core.sensor_data import MotionData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)
    streaming_client = aria.StreamingClient()

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Imu
    config.message_queue_size[aria.StreamingDataType.Imu] = 1000

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    angular_position = [0.0]
    last_imu_time = [None]
    imu_start_time = [None]

    imu_filename = "imu_data_one_hour.csv"
    csv_file = open(imu_filename, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp(s)", "GyroX(rad/s)", "GyroY(rad/s)", "GyroZ(rad/s)"])

    class StreamingClientObserver:
        def on_imu_received(self, data_list: list[MotionData], imu_idx: int):
            if imu_idx == 0:
                for data in data_list:
                    timestamp = data.capture_timestamp_ns / 1e9
                    gyro = data.gyro_radsec

                    if imu_start_time[0] is None:
                        imu_start_time[0] = timestamp

                    writer.writerow([timestamp, *gyro])

                    if last_imu_time[0] is not None:
                        dt = timestamp - last_imu_time[0]
                        delta_angle = gyro[0] * dt
                        angular_position[0] += delta_angle
                    last_imu_time[0] = timestamp

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to IMU data")
    streaming_client.subscribe()

    loop_start_time = time.time()
    while time.time() - loop_start_time < 3600.0:
        print(time.time() - loop_start_time, "seconds elapsed")

    print("Stop listening to data")
    streaming_client.unsubscribe()
    csv_file.close()

    print(f"\nSaved IMU data to {imu_filename}")

if __name__ == "__main__":
    main()
