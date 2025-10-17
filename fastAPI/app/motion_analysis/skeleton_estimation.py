from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np

try:
    from dtw import dtw as dtw_fn
except ImportError:  # pragma: no cover - exercised via runtime fallback
    def dtw_fn(
        sequence_a: np.ndarray,
        sequence_b: np.ndarray,
        dist_method,
    ) -> SimpleNamespace:
        len_a = len(sequence_a)
        len_b = len(sequence_b)
        if len_a == 0 or len_b == 0:
            raise ValueError("Sequences must be non-empty for DTW")

        cost = np.full((len_a + 1, len_b + 1), np.inf, dtype=float)
        cost[0, 0] = 0.0

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                distance = dist_method(sequence_a[i - 1], sequence_b[j - 1])
                cost[i, j] = distance + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        i, j = len_a, len_b
        index1: List[int] = []
        index2: List[int] = []

        while i > 0 and j > 0:
            index1.append(i - 1)
            index2.append(j - 1)
            choices = (
                cost[i - 1, j - 1],
                cost[i - 1, j],
                cost[i, j - 1],
            )
            step = int(np.argmin(choices))
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1

        while i > 0:
            i -= 1
            index1.append(i)
            index2.append(0)
        while j > 0:
            j -= 1
            index1.append(0)
            index2.append(j)

        index1.reverse()
        index2.reverse()

        distance_total = cost[len_a, len_b]
        normalized_distance = distance_total / max(len_a, len_b)

        return SimpleNamespace(
            distance=float(distance_total),
            normalizedDistance=float(normalized_distance),
            index1=np.array(index1, dtype=int),
            index2=np.array(index2, dtype=int),
        )

dtw = dtw_fn

# Reuse global pose helpers to avoid repeated construction.
mp_pose = mp.solutions.pose


def _connection_index(value) -> int:
    return getattr(value, "value", int(value))


_POSE_CONNECTION_IDS = [
    (_connection_index(a), _connection_index(b)) for a, b in mp_pose.POSE_CONNECTIONS
]
_RIGHT_ANKLE_ID = mp_pose.PoseLandmark.RIGHT_ANKLE.value


@dataclass
class PoseFrame:
    frame_id: int
    landmarks: List[Dict[str, float]]


@dataclass
class PoseSequence:
    sequence_id: str
    frames: List[PoseFrame]
    width: int
    height: int
    source_path: str

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def calculate_angle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Calculate the angle between three points in degrees."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    c_arr = np.array(c)
    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - np.arctan2(a_arr[1] - b_arr[1], a_arr[0] - b_arr[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


class SkeletonEstimator:
    """Extract pose landmarks from videos and compare sequences."""

    def __init__(
        self,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._pose_cache: Dict[str, PoseSequence] = {}

    def extract_landmarks(
        self,
        video_path: str,
        *,
        sequence_id: Optional[str] = None,
        store: bool = True,
    ) -> PoseSequence:
        """Run Mediapipe pose on a video and return normalized landmarks per frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if width == 0 or height == 0:
            cap.release()
            raise ValueError(f"Video metadata missing dimensions: {video_path}")

        frames: List[PoseFrame] = []
        frame_id = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                landmarks: List[Dict[str, float]] = []
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        landmarks.append(
                            {
                                "id": idx,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility,
                            }
                        )
                frames.append(PoseFrame(frame_id=frame_id, landmarks=landmarks))
                frame_id += 1

        cap.release()

        seq_id = sequence_id or uuid.uuid4().hex
        sequence = PoseSequence(
            sequence_id=seq_id,
            frames=frames,
            width=width,
            height=height,
            source_path=video_path,
        )

        if store:
            self._pose_cache[seq_id] = sequence
        return sequence

    def get_cached_sequence(self, sequence_id: str) -> Optional[PoseSequence]:
        return self._pose_cache.get(sequence_id)

    def convert_to_angles(self, frames: Iterable[PoseFrame]) -> np.ndarray:
        """Convert a pose sequence to joint-angle feature vectors for DTW."""
        angle_sequences: List[List[float]] = []
        for frame in frames:
            if not frame.landmarks:
                continue
            land_map = {lm["id"]: lm for lm in frame.landmarks}
            try:
                shoulder_l = [land_map[mp_pose.PoseLandmark.LEFT_SHOULDER.value][axis] for axis in ("x", "y")]
                elbow_l = [land_map[mp_pose.PoseLandmark.LEFT_ELBOW.value][axis] for axis in ("x", "y")]
                wrist_l = [land_map[mp_pose.PoseLandmark.LEFT_WRIST.value][axis] for axis in ("x", "y")]
                hip_l = [land_map[mp_pose.PoseLandmark.LEFT_HIP.value][axis] for axis in ("x", "y")]
                knee_l = [land_map[mp_pose.PoseLandmark.LEFT_KNEE.value][axis] for axis in ("x", "y")]
                ankle_l = [land_map[mp_pose.PoseLandmark.LEFT_ANKLE.value][axis] for axis in ("x", "y")]
                shoulder_r = [land_map[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][axis] for axis in ("x", "y")]
                elbow_r = [land_map[mp_pose.PoseLandmark.RIGHT_ELBOW.value][axis] for axis in ("x", "y")]
                wrist_r = [land_map[mp_pose.PoseLandmark.RIGHT_WRIST.value][axis] for axis in ("x", "y")]
                hip_r = [land_map[mp_pose.PoseLandmark.RIGHT_HIP.value][axis] for axis in ("x", "y")]
                knee_r = [land_map[mp_pose.PoseLandmark.RIGHT_KNEE.value][axis] for axis in ("x", "y")]
                ankle_r = [land_map[mp_pose.PoseLandmark.RIGHT_ANKLE.value][axis] for axis in ("x", "y")]
            except KeyError:
                continue

            feature_vector = [
                calculate_angle(shoulder_l, elbow_l, wrist_l),
                calculate_angle(hip_l, shoulder_l, elbow_l),
                calculate_angle(shoulder_l, hip_l, knee_l),
                calculate_angle(hip_l, knee_l, ankle_l),
                calculate_angle(shoulder_r, elbow_r, wrist_r),
                calculate_angle(hip_r, shoulder_r, elbow_r),
                calculate_angle(shoulder_r, hip_r, knee_r),
                calculate_angle(hip_r, knee_r, ankle_r),
            ]
            angle_sequences.append(feature_vector)
        return np.array(angle_sequences)

    def compare_sequences(self, seq_a: PoseSequence, seq_b: PoseSequence) -> Dict[str, object]:
        """Compare two pose sequences with DTW and return similarity metrics."""
        angles_a = self.convert_to_angles(seq_a.frames)
        angles_b = self.convert_to_angles(seq_b.frames)

        if len(angles_a) == 0 or len(angles_b) == 0:
            raise ValueError("Pose data missing for comparison")

        manhattan_distance = lambda x, y: np.abs(x - y).sum()
        alignment = dtw_fn(angles_a, angles_b, dist_method=manhattan_distance)

        normalized_distance = getattr(alignment, "normalizedDistance", None)
        if normalized_distance is None:
            max_len = max(len(angles_a), len(angles_b))
            normalized_distance = alignment.distance / max(1, max_len)

        similarity = 100.0 * np.exp(-0.005 * normalized_distance)

        return {
            "dtw_distance": float(alignment.distance),
            "similarity_percentage": float(similarity),
            "path": {
                "query": list(map(int, alignment.index1)),
                "reference": list(map(int, alignment.index2)),
            },
        }

    def clear_cache(self) -> None:
        self._pose_cache.clear()


def _landmarks_to_points(
    landmarks: List[Dict[str, float]],
    width: int,
    height: int,
) -> Dict[int, Tuple[int, int]]:
    points: Dict[int, Tuple[int, int]] = {}
    for landmark in landmarks:
        x = int(round(landmark["x"] * width))
        y = int(round(landmark["y"] * height))
        points[landmark["id"]] = (x, y)
    return points


def draw_pose_frame(
    frame: PoseFrame,
    *,
    width: int,
    height: int,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
    align_to: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Render a pose frame to an image and optionally align to a reference point."""
    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)

    if not frame.landmarks:
        return canvas, None

    points = _landmarks_to_points(frame.landmarks, width, height)

    for start_id, end_id in _POSE_CONNECTION_IDS:
        start_pt = points.get(start_id)
        end_pt = points.get(end_id)
        if start_pt and end_pt:
            cv2.line(canvas, start_pt, end_pt, connection_color, 2)

    for pt in points.values():
        cv2.circle(canvas, pt, 3, landmark_color, -1)

    anchor = points.get(_RIGHT_ANKLE_ID)
    if align_to and anchor:
        dx = int(align_to[0] - anchor[0])
        dy = int(align_to[1] - anchor[1])
        translation = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, translation, (width, height))

    return canvas, anchor


def render_sequence_frames(
    sequence: PoseSequence,
    *,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
    align_to: Optional[Tuple[int, int]] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], Optional[Tuple[int, int]]]:
    """Render every frame in a pose sequence."""
    rendered: List[np.ndarray] = []
    reference_anchor: Optional[Tuple[int, int]] = None
    width = canvas_size[0] if canvas_size else sequence.width
    height = canvas_size[1] if canvas_size else sequence.height
    for frame in sequence.frames:
        image, anchor = draw_pose_frame(
            frame,
            width=width,
            height=height,
            landmark_color=landmark_color,
            connection_color=connection_color,
            background_color=background_color,
            align_to=align_to,
        )
        if anchor is not None:
            if reference_anchor is None:
                reference_anchor = anchor
        rendered.append(image)
    return rendered, reference_anchor


def encode_video(
    frames: Iterable[np.ndarray],
    *,
    width: int,
    height: int,
    fps: int = 30,
    fourcc: Union[str, Sequence[str]] = ("avc1", "H264", "mp4v"),
) -> bytes:
    """Encode a list of frames into MP4 bytes, preferring browser-friendly codecs."""
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("No frames to encode")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name

        codecs: List[str]
        if isinstance(fourcc, str):
            codecs = [fourcc]
        else:
            codecs = list(fourcc)

        # Ensure we always fall back to mp4v even if callers override the list.
        if "mp4v" not in codecs:
            codecs.append("mp4v")

        writer: Optional[cv2.VideoWriter] = None
        selected_codec: Optional[str] = None
        for codec in codecs:
            writer = cv2.VideoWriter(
                tmp_path,
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if writer.isOpened():
                selected_codec = codec
                break
            writer.release()
            writer = None

        if writer is None or selected_codec is None:
            raise RuntimeError("Failed to initialize video writer with supported codecs")

        for frame in frames_list:
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
