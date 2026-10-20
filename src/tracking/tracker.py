"""
Multi-pedestrian tracker with metric ground-plane association,
tentative confirmation, coasting, and track expiration.
"""

from __future__ import annotations
import logging
import math
from typing import List, Optional
import uuid

from src.core.config import TrackingConfig
from src.core.schemas import Detection, Track, TrackState
from src.tracking.track import PedestrianTrack

logger = logging.getLogger(__name__)


class MultiPedestrianTracker:
    """
    Online multi-pedestrian tracker.
    Uses calibrated ground-plane metric distances and image IoU for association.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks: List[PedestrianTrack] = []
        self._next_id = 1
        self.last_update_sec: Optional[float] = None

    def update(self, detections: List[Detection], timestamp_sec: float) -> List[Track]:
        """
        Associate detections with existing tracks, update states, and return active tracks.
        """
        dt = 0.033
        if self.last_update_sec is not None:
            dt = max(0.001, timestamp_sec - self.last_update_sec)
        self.last_update_sec = timestamp_sec

        matched_tracks = set()
        matched_detections = set()

        # Step 1: Associate existing tracks with new detections
        if self.tracks and detections:
            cost_matrix = []
            for t_idx, track in enumerate(self.tracks):
                row = []
                tx, ty = track.ground_pos_m
                for d_idx, det in enumerate(detections):
                    dx, dy = det.calibrated_ground_pos_m or (0.0, 5.0)
                    dist = math.sqrt((tx - dx)**2 + (ty - dy)**2)
                    iou = track.bbox.iou(det.bbox)

                    # Association score: lower is better
                    # If IoU is high, distance weight is reduced
                    score = dist - (iou * 2.0)
                    row.append(score)
                cost_matrix.append(row)

            # Greedy matching
            # Find minimum score pairs
            pairs = []
            for t_idx in range(len(self.tracks)):
                for d_idx in range(len(detections)):
                    pairs.append((cost_matrix[t_idx][d_idx], t_idx, d_idx))
            pairs.sort(key=lambda x: x[0])

            for score, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_detections:
                    continue

                track = self.tracks[t_idx]
                det = detections[d_idx]
                tx, ty = track.ground_pos_m
                dx, dy = det.calibrated_ground_pos_m or (0.0, 5.0)
                dist = math.sqrt((tx - dx)**2 + (ty - dy)**2)
                iou = track.bbox.iou(det.bbox)

                # Match if within metric distance threshold OR good visual IoU
                if dist <= self.config.distance_threshold_m or iou >= self.config.iou_association_threshold:
                    matched_tracks.add(t_idx)
                    matched_detections.add(d_idx)
                    track.update(det)

                    # State promotion
                    if track.state == TrackState.TENTATIVE and track.hit_streak >= self.config.min_hits_to_confirm:
                        track.state = TrackState.CONFIRMED

        # Step 2: Coast unmatched existing tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.predict_coast(dt)
                if track.missed_frames > self.config.max_missed_frames:
                    track.state = TrackState.DELETED

        # Step 3: Initialize new tentative tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                prefix = (
                    "PED" if det.class_name == "pedestrian"
                    else "HOLE" if det.class_name == "pothole"
                    else "PAPER" if det.class_name == "paper"
                    else "OBS"
                )
                track_id = f"{prefix}_{self._next_id:03d}"
                self._next_id += 1
                new_track = PedestrianTrack(track_id=track_id, initial_detection=det)
                if self.config.min_hits_to_confirm <= 1:
                    new_track.state = TrackState.CONFIRMED
                self.tracks.append(new_track)

        # Step 4: Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        # Return schema representations of active tracks
        return [t.to_schema() for t in self.tracks]
