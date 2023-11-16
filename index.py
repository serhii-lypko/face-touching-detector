import cv2
import mediapipe as mp
import subprocess

# TODO: drop the script execution when OS goes to sleep
# TODO: collect touching occurrences statistics?


class FaceTouchingDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.webcam = cv2.VideoCapture(0)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    @staticmethod
    def turn_off_screen():
        subprocess.run("pmset displaysleepnow", shell=True)

    @staticmethod
    def get_face_boundaries(stream):
        face_landmarks = stream.face_landmarks.landmark

        min_x = min(lm.x for lm in face_landmarks)
        max_x = max(lm.x for lm in face_landmarks)
        min_y = min(lm.y for lm in face_landmarks)
        max_y = max(lm.y for lm in face_landmarks)

        return min_x, max_x, min_y, max_y

    @staticmethod
    def is_point_within_the_face_area(face_boundaries, point):
        face_min_x, face_max_x, face_min_y, face_max_y = face_boundaries
        return (
            face_min_x < point.x < face_max_x and
            face_min_y < point.y < face_max_y
        )

    def is_hand_within_the_face_area(self, stream, hand_landmark):
        face_boundaries = self.get_face_boundaries(stream)
        return any(self.is_point_within_the_face_area(face_boundaries, lm_pt) for lm_pt in hand_landmark)

    def run(self):
        while True:
            _, frame = self.webcam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stream = self.holistic.process(frame)

            if stream.face_landmarks:
                lh_is_touching = stream.left_hand_landmarks and self.is_hand_within_the_face_area(
                    stream, stream.left_hand_landmarks.landmark)
                rh_is_touching = stream.right_hand_landmarks and self.is_hand_within_the_face_area(
                    stream, stream.right_hand_landmarks.landmark)

                if lh_is_touching or rh_is_touching:
                    self.turn_off_screen()

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FaceTouchingDetector()
    detector.run()
