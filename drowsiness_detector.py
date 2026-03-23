import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
import winsound
import threading
import time
from collections import deque
from datetime import datetime

class DrowsinessDetector:
    def __init__(self):
        # MediaPipe Face Mesh init
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # landmark indices
        self.LEFT_EYE     = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE    = [33, 160, 158, 133, 153, 144]
        self.MOUTH_OUTER  = [61, 291]
        self.MOUTH_INNER  = [13, 14, 78, 308]

        # thresholds
        self.EYE_AR_THRESH         = 0.20
        self.EYE_AR_CONSEC_FRAMES  = 20
        self.MOUTH_AR_THRESH       = 0.55
        self.MOUTH_AR_CONSEC_FRAMES= 15
        self.PITCH_THRESH          = -12
        self.YAW_THRESH            = 20
        self.HEAD_CONSEC_FRAMES    = 10

        # state
        self.eye_counter   = 0
        self.mouth_counter = 0
        self.head_counter  = 0
        self.total_alerts  = 0
        self.status        = "Awake"
        self.head_status   = "Normal"
        self.events        = []
        self._beep_active  = False

        # smoothing buffers
        self.ear_buffer = deque(maxlen=10)
        self.mar_buffer = deque(maxlen=5)

    def eye_aspect_ratio(self, pts):
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, outer, inner):
        horiz = dist.euclidean(outer[0], outer[1])
        v1    = dist.euclidean(inner[0], inner[1])
        v2    = dist.euclidean(inner[2], inner[3])
        return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0

    def get_point(self, lm, idx, shape):
        pt = lm.landmark[idx]
        return (int(pt.x * shape[1]), int(pt.y * shape[0]))

    def get_head_pose(self, frame, lm):
        img_pts = np.array([
            self.get_point(lm, 1,   frame.shape),  # nose tip
            self.get_point(lm, 152, frame.shape),  # chin
            self.get_point(lm, 263, frame.shape),  # right eye corner
            self.get_point(lm, 33,  frame.shape),  # left eye corner
            self.get_point(lm, 287, frame.shape),  # right mouth
            self.get_point(lm, 57,  frame.shape),  # left mouth
        ], dtype="double")
        mdl_pts = np.array([
            (0.0,   0.0,    0.0),
            (0.0,  -63.6,  -12.5),
            (43.3,  32.7,  -26.0),
            (-43.3, 32.7,  -26.0),
            (28.9, -28.9,  -24.1),
            (-28.9,-28.9,  -24.1)
        ])
        h, w = frame.shape[:2]
        cam_mtx = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype="double")
        dist_coeffs = np.zeros((4,1))
        ok, rvec, tvec = cv2.solvePnP(mdl_pts, img_pts, cam_mtx, dist_coeffs)
        if not ok:
            return "Normal"
        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _,_,_,_,_,_,eul = cv2.decomposeProjectionMatrix(proj)
        pitch, yaw, _ = [float(x) for x in eul]
        if pitch < self.PITCH_THRESH:
            return "Down"
        if yaw > self.YAW_THRESH:
            return "Turned Right"
        if yaw < -self.YAW_THRESH:
            return "Turned Left"
        return "Normal"

    def _beep_loop(self):
        while self._beep_active:
            winsound.Beep(2000, 200)
            time.sleep(0.1)
            winsound.Beep(1500, 200)
            time.sleep(0.3)

    def detect_drowsiness(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        eye_d = mouth_d = head_d = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]

            # EAR (eye closing)
            left  = [self.get_point(lm, i, frame.shape) for i in self.LEFT_EYE]
            right = [self.get_point(lm, i, frame.shape) for i in self.RIGHT_EYE]
            ear   = (self.eye_aspect_ratio(left) + self.eye_aspect_ratio(right)) / 2
            self.ear_buffer.append(ear)
            smooth_ear = sum(self.ear_buffer)/len(self.ear_buffer)
            
            if smooth_ear < self.EYE_AR_THRESH:
                self.eye_counter += 1
                progress = min(self.eye_counter / self.EYE_AR_CONSEC_FRAMES * 100, 100)
                cv2.putText(frame, f"Drowsiness: {progress:.0f}%", (10,250),
                           cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255),2)
            else:
                self.eye_counter = max(0, self.eye_counter - 1)
            eye_d = self.eye_counter >= self.EYE_AR_CONSEC_FRAMES
            
            for points in [left, right]:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(frame, [pts], True, (0,255,0), 1)
                
            ear_bar_length = int(smooth_ear * 100)
            cv2.rectangle(frame, (10, 320), (10 + ear_bar_length, 340), (0,255,0), -1)
            cv2.putText(frame, f"EAR: {smooth_ear:.2f}", (10,310),
                       cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255),2)

            # MAR (yawning)
            outer = [self.get_point(lm, i, frame.shape) for i in self.MOUTH_OUTER]
            inner = [self.get_point(lm, i, frame.shape) for i in self.MOUTH_INNER]
            mar   = self.mouth_aspect_ratio(outer, inner)
            self.mar_buffer.append(mar)
            smooth_mar = sum(self.mar_buffer)/len(self.mar_buffer)

            # debug draw & log
            print(f"[MAR] {smooth_mar:.2f}")
            for p in outer + inner:
                cv2.circle(frame, p, 2, (0,0,255), -1)

            if smooth_mar > self.MOUTH_AR_THRESH:
                self.mouth_counter += 1
            else:
                self.mouth_counter = 0
            mouth_d = self.mouth_counter >= self.MOUTH_AR_CONSEC_FRAMES

            # Head pose
            self.head_status = self.get_head_pose(frame, lm)
            if self.head_status != "Normal":
                self.head_counter += 1
            else:
                self.head_counter = 0
            head_d = self.head_counter >= self.HEAD_CONSEC_FRAMES

            # overlays
            cv2.putText(frame, f"Head: {self.head_status}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255,0,0),2)

            if eye_d:
                cv2.putText(frame, "Eyes Closed", (10,130),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255),2)
            if mouth_d:
                cv2.putText(frame, "Yawning", (10,160),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255),2)
            if head_d:
                cv2.putText(frame, f"Looking {self.head_status}", (10,190),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255),2)

        # overall status
        if eye_d:
            new_status = "Drowsy!"
        else:
            new_status = "Awake"

        if new_status != self.status:
            self.status = new_status
            if new_status == "Drowsy!":
                self.total_alerts += 1
                self.events.append(datetime.now().isoformat())
                self._beep_active = True
                threading.Thread(target=self._beep_loop, daemon=True).start()
            else:
                self._beep_active = False

        color = (0,0,255) if self.status == "Drowsy!" else (0,255,0)
        cv2.putText(frame, f"Status: {self.status}", (10,220),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, color,2)

        return frame, self.status

    def get_status(self):
        return {
            "eyes_closed":  self.eye_counter   >= self.EYE_AR_CONSEC_FRAMES,
            "yawning":      self.mouth_counter >= self.MOUTH_AR_CONSEC_FRAMES,
            "head_down":    self.head_status   == "Down" and self.head_counter >= self.HEAD_CONSEC_FRAMES,
            "head_left":    self.head_status   == "Turned Left" and self.head_counter >= self.HEAD_CONSEC_FRAMES,
            "head_right":   self.head_status   == "Turned Right" and self.head_counter >= self.HEAD_CONSEC_FRAMES,
            "overall":      self.status,
            "total_alerts": self.total_alerts,
            "events":       self.events
        }

    def get_events(self):
        return self.events
