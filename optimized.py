import time
import subprocess

import cv2
import mediapipe as mp

# Lightweight face-touch detector.
#
# Why this is cheaper than index.py (which used mp.solutions.Holistic):
#   1. Holistic runs FaceMesh (468 landmarks) + Pose + 2 hand models on EVERY
#      frame. We don't need any of that precision. We only need a face bounding
#      box and hand positions, so we use the much lighter FaceDetection
#      (BlazeFace) + Hands(model_complexity=0, the "lite" model).
#   2. The original loop was uncapped (`while True` with no sleep), so it pinned
#      a CPU core at 100% and ran inference as fast as possible -> heat. Here we
#      cap to TARGET_FPS; hand-to-face contact does not need 30-60 fps.
#   3. We downscale the frame before inference. Inference cost scales with pixel
#      count; a 480px-wide frame is plenty for this task.
#   4. Face detection is cheap, hand detection is not. We run the face detector
#      first and ONLY run the hand model when a face is actually visible. When
#      you're not in frame, we skip the expensive step entirely.

TARGET_FPS = 6           # frames actually processed per second
PROCESS_WIDTH = 480      # frame is downscaled to this width before inference
FACE_PADDING = 0.05      # expand face box by 5% on each side (forgiveness)
TRIGGER_COOLDOWN = 3.0   # seconds to wait after sleeping the display


class FaceTouchingDetector:
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)
        # Ask the camera for a modest resolution so we move/decode fewer pixels.
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,            # 0 = lite model, the key saving
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.last_trigger = 0.0

    @staticmethod
    def turn_off_screen():
        subprocess.run("pmset displaysleepnow", shell=True)

    @staticmethod
    def get_face_box(detection):
        box = detection.location_data.relative_bounding_box
        min_x = box.xmin - FACE_PADDING
        max_x = box.xmin + box.width + FACE_PADDING
        min_y = box.ymin - FACE_PADDING
        max_y = box.ymin + box.height + FACE_PADDING
        return min_x, max_x, min_y, max_y

    @staticmethod
    def hand_in_face(face_box, hand_landmarks):
        min_x, max_x, min_y, max_y = face_box
        return any(
            min_x < lm.x < max_x and min_y < lm.y < max_y
            for lm in hand_landmarks.landmark
        )

    def process(self, rgb):
        face_result = self.face_detection.process(rgb)
        if not face_result.detections:
            return  # no face -> skip the expensive hand model entirely

        hand_result = self.hands.process(rgb)
        if not hand_result.multi_hand_landmarks:
            return

        # Compare against the largest detected face box.
        face_box = max(
            (self.get_face_box(d) for d in face_result.detections),
            key=lambda b: (b[1] - b[0]) * (b[3] - b[2]),
        )

        for hand_landmarks in hand_result.multi_hand_landmarks:
            if self.hand_in_face(face_box, hand_landmarks):
                now = time.time()
                if now - self.last_trigger > TRIGGER_COOLDOWN:
                    self.turn_off_screen()
                    self.last_trigger = now
                return

    def run(self):
        frame_interval = 1.0 / TARGET_FPS
        try:
            while True:
                start = time.time()

                ok, frame = self.webcam.read()
                if not ok:
                    time.sleep(frame_interval)
                    continue

                scale = PROCESS_WIDTH / frame.shape[1]
                if scale < 1.0:
                    frame = cv2.resize(
                        frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False  # lets mediapipe avoid a copy
                self.process(rgb)

                # Throttle: sleep off the remainder of the frame budget so we
                # don't spin the CPU at 100%.
                elapsed = time.time() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            self.webcam.release()
            self.face_detection.close()
            self.hands.close()


if __name__ == "__main__":
    FaceTouchingDetector().run()
