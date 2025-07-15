# Imports
## standard library imports
import argparse
import sys
import time
import csv
import threading
import io
import math
import os
import webbrowser
from collections import defaultdict
from multiprocessing import Process, Queue

## third-party imports
import cv2
import pyttsx3
import openai
import numpy as np
import speech_recognition as sr
import torch
from ultralytics import YOLO
from dotenv import load_dotenv
from pydub import AudioSegment
from flask import Flask, request, render_template

## aria imports
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData
from projectaria_tools.core import data_provider
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.stream_id import StreamId
from common import quit_keypress, update_iptables

## local imports
from support_functions.meal_knowledge import can_make_meal, meals
from support_functions.action_recognition import get_stove_border_frame, overlaps_with_stove_frame, live_predict
from support_functions.looking_at import get_gazed_object
from inference import infer # Given by Aria


# Load environment variables
load_dotenv()
## get the OpenAI API key from the environment variable
open_ai_api_key = os.getenv("OPENAI_API_KEY")

# Load the YOLO model
model = YOLO('image_model/yolov8n_hands_0.pt')

# Global state
## States for object detection and position tracking
angular_position = [0.0] # Initial angle in radians
last_imu_time = [None] # Last IMU timestamp in seconds
object_positions = {} # Dictionary to store object positions with class names
gyro_bias_x = 0.004756912266351248 # bias of the x-axis gyro in rad/sec

## Queue for text-to-speech messages
tts_queue = None

## Initial chat history with system prompt
chat_history = [
    {"role": "system", "content": "You're a helpful assistant for a person with dementia. Use object detection data and support step-by-step meal help. Speak simply and clearly. Do not ask me for more information, just use the items that get sent to you, do not assume anything else."}
]

## States for stove information
stove_state = {
    "last_overlap": False,
    "overlap_start_time": None,
    "toggle_count": 0,
    "has_toggled_once": False
} # tracking stove state
stove_turned_on_time = None # how long the stove has been turned on
stove_reminder_count = 1 # number of times reminder has been sent
stove_reminder_frequency_sec = 60 # definition of how often the reminder should be sent in seconds

## States of faucet
faucet_turned_on_time = None
faucet_reminder_count = 1
faucet_reminder_interval_sec = 10

## variables for tracking how often objects are detected 
detection_persistence = defaultdict(int) # saves in how many consecutive frames an object has been detected
eligible_objects = set() # keeps the objects that have been detected for a certain number of consecutive frames

## Variables for eye tracking
provider = data_provider.create_vrs_data_provider("reference_vrs/Profile_18.vrs")
device_calibration = provider.get_device_calibration()
rgb_stream_id = StreamId("214-1")
rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
got_rgb_image = False

## Variables for gaze target
gaze_target_label = [None]

## Faucet state
faucet_on = False # whether the faucet is currently on

# Flask setup
app = Flask(__name__)
web_object_positions = {}


@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history, tts_queue

    response_text = ""

    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if user_input:
            # Add current object/direction summary to the user message
            object_summary = get_object_direction_summary()

            available_items = set(object_positions.keys())
            available_tools = {item for item in available_items if item in {"Cup", "Faucet", "Plate", "Pot", "Spoon", "Stove"}}
            available_ingredients = available_items - available_tools

            possible_meals = [
                meal for meal in meals
                if can_make_meal(meal, available_ingredients, available_tools)
            ]

            meals_summary = ", ".join(possible_meals) if possible_meals else "Nothing can be made with the items I have"

            full_input = (f"The detected objects in my surroundings right now are: {object_summary}. "
                          f"With those items I can make this: {meals_summary}. "
                          f"Besides of that I am currently looking at: {gaze_target_label[0]}. "
                          f"Do not suggest anything else to me. {user_input}"
                          )

            chat_history.append({"role": "user", "content": full_input})

            try:
                client = openai.OpenAI(api_key=open_ai_api_key)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=chat_history,
                    max_tokens=500,
                    temperature=0.7
                )

                assistant_reply = response.choices[0].message.content
                chat_history.append({"role": "assistant", "content": assistant_reply})
                response_text = assistant_reply

                if tts_queue:
                    tts_queue.put(assistant_reply)

            except Exception as e:
                response_text = f"Error communicating with ChatGPT: {e}"

    return render_template("index.html", response=response_text, history=chat_history[1:])

@app.route("/suggest", methods=["POST"])
def suggest_meal():
    global chat_history, tts_queue

    item_names = object_positions.keys()
    prompt, idea = get_chatgpt_ideas(item_names, chat_history)

    # Only append if prompt not already in chat_history
    if not any(entry.get("content") == prompt for entry in chat_history if entry.get("role") == "user"):
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": idea})

    if tts_queue:
        tts_queue.put(idea)

    return render_template("index.html", response=idea, history=chat_history[1:])


@app.route("/transcribe_voice", methods=["POST"])
def transcribe_voice():
    if "audio" not in request.files:
        return {"text": ""}

    uploaded_file = request.files["audio"]
    audio_bytes = uploaded_file.read()

    try:
        # Decode from webm
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return {"text": text}
    except Exception as e:
        print("Voice transcription failed:", e)
        return {"text": ""}

## run the Flask app on Port 5000
def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)

## Open the web interface automatically
webbrowser.open("http://127.0.0.1:5000")

# Functions
## image recognition function
def rec_image(image, model, allowed_classes=None):
    results = model(image)
    detections = []

    for result in results:
        for box in result.boxes:
            # only accept boxes with a confidence above 0.4
            if box.conf[0].item() < 0.4:
                continue

            cls = int(box.cls[0].item())
            class_name = model.names[cls]

            if allowed_classes is not None:
                if isinstance(allowed_classes[0], str) and class_name not in allowed_classes:
                    continue
                elif isinstance(allowed_classes[0], int) and cls not in allowed_classes:
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            image_width = image.shape[1]

            conf = box.conf[0].item()
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append({
                "class_name": class_name,
                "center_x": center_x,
                "image_width": image_width,
                "bbox": (x1, y1, x2, y2)
            })

    return image, detections

## idea generation function
def get_chatgpt_ideas(item_names, chat_history):
    available_items = set(item_names)
    # available_tools = {item for item in available_items if item in {"oven", "knife", "cutting board", "pan", "sink", "cup"}}
    available_tools = {item for item in available_items if item in {"Cup", "Faucet", "Plate", "Pot", "Spoon", "Stove"}}
    available_ingredients = available_items - available_tools

    possible_meals = [
        meal for meal in meals
        if can_make_meal(meal, available_ingredients, available_tools)
    ]

    object_summary = get_object_direction_summary()
    meals_summary = ", ".join(possible_meals) if possible_meals else "none"

    prompt = (
        f"Imagine I'm a person with dementia. Don't mention that I have dementia. "
        f"I have the following items with the information where they are right now, nothing else: {object_summary}. "
        f"Given this and assuming only the following things are possible: {meals_summary}, "
        f"can you give me steps for how to do only one of those listed things with the items I have?"
        f"If the list of things that are possible states \"none\", tell me that nothing is possible to make."
    )

    full_chat_history = chat_history + [
        {"role": "user", "content": prompt}
    ]

    try:
        client = openai.OpenAI(api_key=open_ai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_chat_history,
            max_tokens=500,
            temperature=0.7,
        )
        return prompt, response.choices[0].message.content

    except Exception as e:
        return prompt, f"Error from ChatGPT: {e}"

## Text-to-Speech worker function
def tts_worker(queue: Queue):
    engine = pyttsx3.init()
    while True:
        text = queue.get()
        if text == "__QUIT__":
            break
        engine.say(text)
        engine.runAndWait()

## Argument parser for command line options defined by Aria SDK
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_iptables", 
                        default=False, 
                        action="store_true",
                        help="Update iptables to enable receiving the data stream, only for Linux.")
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=f"{os.path.dirname(__file__)}/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth",
        help="location of the model weights",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=f"{os.path.dirname(__file__)}/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml",
        help="location of the model config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run inference on",
    )
    return parser.parse_args()

## object position encoding
### Normalize angle to be within -pi to pi
def normalize_angle(angle):
    while angle >= math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

### Calculate relative angle between person and target object
def relative_angle(person_angle, target_angle):
    return normalize_angle(target_angle - person_angle)

### Determine the zone based on the relative angle
def get_zone(rel_angle):
    if -0.25*math.pi <= rel_angle < 0.25*math.pi:
        return "front"
    elif 0.25*math.pi <= rel_angle < 0.75*math.pi:
        return "right"
    elif rel_angle >= 0.75*math.pi or rel_angle < -0.75*math.pi:
        return "behind"
    elif -0.75*math.pi <= rel_angle < -0.25*math.pi:
        return "left"

### Get a summary of object directions in zones from get_zone relative to the person
def get_object_direction_summary():
    summary = []
    for label, angle in web_object_positions.items():
        if label == "Hand":
            continue
        rel = relative_angle(angular_position[0], angle)
        zone = get_zone(rel)
        summary.append(f"{label}: {zone}")
    if not summary:
        return "No objects detected."
    else:
        return ", ".join(summary)

# Classes
## StreamingClientObserver to handle incoming data from the Aria glasses
class StreamingClientObserver:
    def __init__(self):
        self.images = {}

    ### Callback for receiving images defined by the Aria SDK
    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

    ### Callback for receiving IMU data 
    def on_imu_received(self, data_list: list[MotionData], imu_idx: int):
        #### only use the first IMU
        if imu_idx == 0:
            for data in data_list:
                gyro = data.gyro_radsec
                #### convert from ns to seconds
                timestamp = data.capture_timestamp_ns / 1e9

                #### update the angular position based on the gyro data
                if last_imu_time[0] is not None:
                    #### calculate the time difference since the last IMU reading
                    dt = timestamp - last_imu_time[0]
                    #### Correct the gyro data by subtracting the bias
                    corrected_gyro_x = gyro[0] - gyro_bias_x
                    #### Calculate the change in angle
                    delta_angle = corrected_gyro_x * dt
                    #### update the current angular position
                    angular_position[0] += delta_angle

                last_imu_time[0] = timestamp

# Main function
def main():
    global tts_queue, stove_turned_on_time, got_rgb_image, gaze_target_label, faucet_on
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    ## Given by Aria SDK
    aria.set_log_level(aria.Level.Info)
    streaming_client = aria.StreamingClient()

    ## configure subscription to listen to Arias RGB and IMU streams (given by Aria SDK)
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.Imu | aria.StreamingDataType.EyeTrack
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.Imu] = 1000
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    ## Set the security options (given by Aria SDK)
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    ## Create and attach observer (given by Aria SDK)
    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    ## Initialize the TTS worker process
    tts_queue = Queue()
    tts_process = Process(target=tts_worker, args=(tts_queue,), daemon=True)
    tts_process.start()
    tts_queue.put("Initializing speech system.")

    ## Subscribe to the streaming client (given by Aria SDK)
    streaming_client.subscribe()

    ## define the RGB window for displaying the images
    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    ## Define allowed classes (TODO: Change, since the new model only has these classes)
    allowed_classes = [
        "Cup", "Faucet", "Pasta", "Plate", "Pot", "Rice", "Spoon", "Stove", "Tea", "Hand"
    ]

    ## get the inference model for eye tracking
    inference_model = infer.EyeGazeInference(
            args.model_checkpoint_path, args.model_config_path, args.device
        )

    ## main loop to process data
    while not quit_keypress():
        if aria.CameraId.Rgb in observer.images:
            ## set that we got a RGB image
            got_rgb_image = True

            ## Make the RGB image ready for processing
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1) # Rotate the image to match the display orientation
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB

            ## get the origin of the RGB image
            orig_h, orig_w = rgb_image.shape[:2]

            ## detect objects in the RGB image
            processed_image, detections = rec_image(rgb_image, model, allowed_classes)

            ## Stove state management
            ### Find Hand and Stove boxes
            hand_boxes = [d["bbox"] for d in detections if d["class_name"] == "Hand"]
            stove_boxes = [d["bbox"] for d in detections if d["class_name"] == "Stove"]

            ### check if user uses stove
            overlapping = False
            if stove_boxes:
                ### Get border frame for the stove (where controls are located)
                outer, inner = get_stove_border_frame(stove_boxes[0])

                ### Check if any hand box overlaps with the stove frame
                for hand_box in hand_boxes:
                    if overlaps_with_stove_frame(hand_box, outer, inner):
                        overlapping = True
                        break
            
            ### Save current time
            current_time = time.time()

            ### check if stove gets turned on or off
            if overlapping:
                if not stove_state["last_overlap"]:
                    stove_state["overlap_start_time"] = current_time
                    stove_state["has_toggled_once"] = False
                    stove_state["last_overlap"] = True
                else:
                    duration = current_time - stove_state["overlap_start_time"]

                    if duration >= 1.5 and not stove_state["has_toggled_once"]:
                        stove_state["toggle_count"] += 1
                        action = "turned on" if stove_state["toggle_count"] % 2 == 1 else "turned off"
                        stove_state["has_toggled_once"] = True
                        if tts_queue:
                            tts_queue.put(f"The stove is {action}")
                        if action == "turned on":
                            stove_turned_on_time = current_time
                        else:
                            stove_turned_on_time = None

                    elif duration >= 5 and stove_state["has_toggled_once"]:
                        stove_state["toggle_count"] += 1
                        action = "turned on" if stove_state["toggle_count"] % 2 == 1 else "turned off"
                        stove_state["has_toggled_once"] = False
                        stove_state["overlap_start_time"] = current_time
                        if tts_queue:
                            tts_queue.put(f"The stove is {action}")
                        if action == "turned on":
                            stove_turned_on_time = current_time
                        else:
                            stove_turned_on_time = None            
            else:
                stove_state["last_overlap"] = False
                stove_state["overlap_start_time"] = None
                stove_state["has_toggled_once"] = False
            
            ### start timer if stove is turned on to speak out reminders to check it
            if stove_turned_on_time is not None:
                if current_time - stove_turned_on_time > stove_reminder_frequency_sec:
                    time_since_on_min = int((stove_reminder_count * stove_reminder_frequency_sec)/60)
                    if tts_queue:
                        if time_since_on_min == 1:
                            tts_queue.put(f"The stove has been turned on for {time_since_on_min} minute. You should check on it.")
                        else:
                            tts_queue.put(f"The stove has been turned on for {time_since_on_min} minutes. You should check on it.")
                    stove_turned_on_time = current_time 
                    stove_reminder_count += 1
            else:
                stove_reminder_count = 1


            ## IFOV = 0.038 deg/pixel => radians per pixel
            IFOV_RAD_PER_PIXEL = np.deg2rad(0.038)

            ## Track detections and update eligibility
            current_frame_objects = set(d["class_name"] for d in detections)

            for class_name in allowed_classes:
                if class_name in current_frame_objects:
                    detection_persistence[class_name] += 1
                else:
                    detection_persistence[class_name] = 0  # reset if not seen in current frame

                if detection_persistence[class_name] >= 10:
                    eligible_objects.add(class_name) 

            ## Update positions for all eligible objects
            for class_name in eligible_objects:
                ### Find current detection for this object (if available)
                matching = [d for d in detections if d["class_name"] == class_name]
                if matching:
                    center_x = matching[0]["center_x"]
                    image_width = matching[0]["image_width"]
                    pixel_offset = center_x - (image_width / 2)
                    angular_offset = pixel_offset * IFOV_RAD_PER_PIXEL
                    adjusted_angle = angular_position[0] + angular_offset
                    object_positions[class_name] = adjusted_angle


            with open("detected_positions.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Object Class", "Angular Position (rad)"])
                for cls, angle in object_positions.items():
                    writer.writerow([cls, angle])

            del observer.images[aria.CameraId.Rgb]
        
        if aria.CameraId.EyeTrack in observer.images and got_rgb_image:
            eye_image = observer.images[aria.CameraId.EyeTrack]

            if eye_image.dtype != np.uint8:
                eye_image = cv2.normalize(eye_image, None, 0, 255, cv2.NORM_MINMAX)
                eye_image = eye_image.astype(np.uint8)

            if len(eye_image.shape) == 2:
                eye_image = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2BGR)

            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(eye_image).to(dtype=torch.uint8)

            preds, lower, upper = inference_model.predict(img)

            eye_gaze = EyeGaze
            eye_gaze.yaw = preds[0][0]
            eye_gaze.pitch = preds[0][1]
            print(f"Yaw: {eye_gaze.yaw}, Pitch: {eye_gaze.pitch}")

            gaze_projection = get_gaze_vector_reprojection(
                    eye_gaze,
                    rgb_stream_label,
                    device_calibration,
                    rgb_camera_calibration,
                    1,
            )

            gaze_x, gaze_y = gaze_projection
            rotated_gaze_x = orig_h - gaze_y
            rotated_gaze_y = gaze_x
            gaze_point = (rotated_gaze_x, rotated_gaze_y)
            gaze_target = get_gazed_object(gaze_point, detections, threshold=1000)
            gaze_target_label[0] = gaze_target
            cv2.circle(processed_image, (int(rotated_gaze_x), int(rotated_gaze_y)), 10, (0, 0, 255), -1)

            del observer.images[aria.CameraId.EyeTrack]

        if got_rgb_image:
            cv2.imshow(rgb_window, processed_image)
            got_rgb_image = False
        
        print(f"The faucet is turned {'on' if faucet_on else 'off'}.")

    # dump chat_history to a file
    with open("chat_history.txt", "w") as f:
        for entry in chat_history:
            f.write(f"{entry['role']}: {entry['content']}\n")

    print("Stop listening to data")
    streaming_client.unsubscribe()
    tts_queue.put("__QUIT__")

if __name__ == "__main__":
    def flask_thread():
        global web_object_positions
        while True:
            web_object_positions = object_positions.copy()
            time.sleep(1)
    
    def faucet_thread():
        global faucet_on, faucet_turned_on_time, faucet_reminder_count
        while True:
            is_on = live_predict() == "faucet"

            if is_on and not faucet_on:
                # Faucet just turned on
                faucet_turned_on_time = time.time()
                faucet_reminder_count = 1
                faucet_on = True

            elif not is_on and faucet_on:
                # Faucet just turned off
                faucet_on = False
                faucet_turned_on_time = None
                faucet_reminder_count = 1

            elif faucet_on and faucet_turned_on_time:
                elapsed = time.time() - faucet_turned_on_time
                if elapsed >= faucet_reminder_count * 10:
                    if tts_queue:
                        tts_queue.put(f"The faucet has been turned on for a bit. You should check on it.")
                    faucet_reminder_count += 1

    # Start Flask and TTS in separate threads
    threading.Thread(target=flask_thread, daemon=True).start()
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=faucet_thread, daemon=True).start()

    main()
