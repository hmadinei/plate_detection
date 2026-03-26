from __future__ import annotations
import cv2
import numpy as np
import torch
from math import atan2, degrees
from ultralytics.engine.results import Results

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import math

from utils import *

def visualize_alignment(
        self,
        image: np.ndarray,
        detections: List[Detection],
        alignment: AlignmentResult,
        color_det=(0, 255, 0),
        color_slots=(255, 0, 0),
) -> np.ndarray:
    """
    Draw detections and slot centers on the image for debugging.
    - Detections: green boxes with char+conf
    - Slot centers: blue vertical lines with slot index
    - Assigned chars: red text at slot center
    """
    vis = image.copy()

    # draw detections
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_det, 1)
        cv2.putText(vis, f"{d.char}:{d.char_conf:.2f}", (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_det, 1)

    # draw slot centers
    H, W = vis.shape[:2]
    for j, cx in enumerate(alignment.expected_centers):
        cx = int(cx)
        cv2.line(vis, (cx, 0), (cx, H), color_slots, 1)
        cv2.putText(vis, str(j), (cx - 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_slots, 1)

    # draw assigned characters on slots
    for r in alignment.sequence:
        if r.char != "?":
            cx = int(alignment.expected_centers[r.index])
            cv2.putText(vis, r.char, (cx - 8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (x1,y1,x2,y2) in rectified coords
    class_probs: Dict[str, float]  # e.g. {'0':0.02, ..., 'A':0.01, ...}
    det_score: float = 1.0  # detector score, optional
    # Derived
    x: float = field(init=False)
    w: float = field(init=False)
    char: str = field(init=False)
    char_conf: float = field(init=False)
    p_digit: float = field(init=False)
    p_letter: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.x = 0.5 * (x1 + x2)
        self.w = max(1e-6, abs(x2 - x1))
        # normalize probs in case they don't sum to 1
        s = sum(self.class_probs.get(k, 0.0) for k in ALL_CHARS)
        probs = {k: (self.class_probs.get(k, 0.0) / s) if s > 0 else 0.0 for k in ALL_CHARS}
        # print('probs in __post_init__ : ',probs)
        self.class_probs = probs
        self.char = max(probs, key=lambda k: probs[k]) if probs else "?"
        self.char_conf = probs.get(self.char, 0.0)
        self.p_digit = sum(probs.get(k, 0.0) for k in DIGITS)
        self.p_letter = sum(probs.get(k, 0.0) for k in LETTERS)


@dataclass
class Slot:
    index: int
    expected_x: float
    slot_type: str  # 'digit' or 'letter'


@dataclass
class SlotResult:
    index: int
    char: str
    conf: float
    assigned_detection: Optional[int]  # index into original detections list, or None
    bbox: Optional[Tuple[float, float, float, float]]


@dataclass
class AlignmentResult:
    sequence: List[SlotResult]
    expected_centers: List[float]
    alignment_logprob: float
    total_cost: float
    used_detections: List[Optional[int]]  # per-slot det index or None


# ----------------------------- Core Aligner ---------------------------------

class PlateAligner:
    def __init__(
            self,
            alpha: float = 1.15,  # scale from median digit width -> advance
            th_letter: float = 0.6,  # threshold to trust letter anchor
            th_edge_drop: float = 0.6,  # drop low-conf edge chars and retry
            lam_x: float = 1.0,
            lam_c: float = 2.0,
            lam_s: float = 1.0,
            lam_g: float = 1.5,
    ):
        self.alpha = alpha
        self.th_letter = th_letter
        self.th_edge_drop = th_edge_drop
        self.lam_x = lam_x
        self.lam_c = lam_c
        self.lam_s = lam_s
        self.lam_g = lam_g

    # # ---- Advance estimation ----
    # def estimate_advance(self, dets: List[Detection]) -> float:
    #     digits = [d.w for d in dets if d.p_digit > 0.5]
    #     pool = digits if digits else [d.w for d in dets]
    #     if not pool:
    #         print('digits not found')
    #         return 10.0  # fallback arbitrary units
    #     w_med = float(np.median(pool))
    #     # todo self alpha dynamicaly
    #     return max(1e-3, w_med * self.alpha)

    # ---- Advance estimation ----
    def estimate_advance(self, dets: List[Detection]) -> float:
        if not dets:
            return 10.0  # fallback arbitrary units

        # Step 1: median width of digits (fallback: all boxes)
        digits = [d.w for d in dets if d.p_digit > 0.5]
        pool = digits if digits else [d.w for d in dets]
        w_med = float(np.median(pool))

        # Step 2: find nearest-neighbor distances (sorted by x)
        dets_sorted = sorted(dets, key=lambda d: d.x)
        neighbor_dists = [
            dets_sorted[i + 1].x - dets_sorted[i].x
            for i in range(len(dets_sorted) - 1 - 1)
        ]  ## todo add the last -1 in documentation

        # Step 3: take minimum neighbor distance (if any)
        min_dist = min(neighbor_dists) if neighbor_dists else None

        # Step 4: compare with threshold
        base_estimate = w_med * self.alpha
        if min_dist is not None and min_dist < w_med * (6 / 5):
            print(f'w_med = {w_med} but return = {max(1e-3, min_dist)} ...(1)')
            return max(1e-3, min_dist)  # use tiny neighbor gap instead
        else:
            print(f'w_med = {w_med} and also it is retrun value! ...(2)')
            return max(1e-3, base_estimate)

    # ---- Build expected x-centers for slots ----
    # x_slots = self.expected_centers_from_anchor(detections[anchor].x, a)
    # x_slots2 = self.expected_centers_from_anchor(detections[anchor].x, a2)
    # x_slots = self.expected_centers_from_offset(ref_x, a, ref_slot)
    def expected_centers_from_anchor(self, anchor_x: float, a: float) -> List[float]:
        x_slots = self.compute_slot_centers(LETTER_INDEX, anchor_x, a)
        return x_slots
        # return [anchor_x + (i - LETTER_INDEX) * a for i in range(8)]

    def expected_centers_from_offset(self, ref_x: float, a: float, ref_slot: int) -> List[float]:
        x_slots = self.compute_slot_centers(ref_slot, ref_x, a)
        return x_slots
        # return [ref_x + (i - ref_slot) * a for i in range(8)]

    def compute_slot_centers(self, anchor_slot: int, anchor_x: float, a: float) -> List[float]:
        """Compute expected centers with slot-specific advance factors."""
        x_slots = [0.0] * 8
        x_slots[anchor_slot] = anchor_x
        # forward
        for j in range(anchor_slot + 1, 8):
            x_slots[j] = x_slots[j - 1] + a * ADV_FACTORS[j]
        # backward
        for j in range(anchor_slot - 1, -1, -1):
            x_slots[j] = x_slots[j + 1] - a * ADV_FACTORS[j + 1]
        return x_slots

    # ---- Type penalty ----
    def type_penalty(self, d: Detection, slot_type: str, letter_unknown_mode: bool) -> float:
        if slot_type == 'letter':
            if d.p_letter >= 0.5:
                return 0.0
            return 1.0 if letter_unknown_mode else 3.0
        else:  # digit slot
            return 0.0 if d.p_digit >= 0.5 else 1.0

    # ---- Dynamic programming alignment with gaps ----
    def align_with_dp(
            self,
            dets_sorted_idx: List[int],
            dets: List[Detection],
            x_slots: List[float],
            slot_types: List[str],
            a: float,
            letter_unknown_mode: bool,
    ) -> Tuple[List[Tuple[str, Optional[int]]], float, List[List[Any]]]:
        """
        Returns:
          path: list of decisions per slot j: ('gap', None) or ('assign', det_index)
          total_cost: float
          dp: the DP table for optional introspection
        """
        # print('detection in align with dp : ',dets)
        # print('-'*10)
        m = len(dets_sorted_idx)
        S = len(x_slots)
        INF = 1e18
        # dp[i][j] = (cost, prev_i, prev_j, decision)
        dp_cost = [[INF] * (S + 1) for _ in range(m + 1)]
        dp_prev = [[None] * (S + 1) for _ in range(m + 1)]
        dp_dec = [[None] * (S + 1) for _ in range(m + 1)]
        dp_cost[0][0] = 0.0

        for i in range(m + 1):
            for j in range(S + 1):
                base = dp_cost[i][j]
                if base >= INF:
                    continue
                # Option A: gap at slot j -> j+1
                if j < S:
                    cost_gap = base + self.lam_g
                    if cost_gap < dp_cost[i][j + 1]:
                        dp_cost[i][j + 1] = cost_gap
                        dp_prev[i][j + 1] = (i, j)
                        dp_dec[i][j + 1] = ('gap', None)
                # Option B: assign detection i -> slot j (monotone)
                if i < m and j < S:
                    di = dets_sorted_idx[i]
                    d = dets[di]
                    pos_term = self.lam_x * abs(d.x - x_slots[j]) / max(1e-6, a)
                    type_term = self.lam_c * self.type_penalty(d, slot_types[j], letter_unknown_mode)
                    conf_term = -self.lam_s * math.log(max(1e-6, d.char_conf))
                    total = base + pos_term + type_term + conf_term
                    if total < dp_cost[i + 1][j + 1]:
                        dp_cost[i + 1][j + 1] = total
                        dp_prev[i + 1][j + 1] = (i, j)
                        dp_dec[i + 1][j + 1] = ('assign', di)

        # choose best i* at j=S
        best_i = min(range(m + 1), key=lambda i: dp_cost[i][S])
        total_cost = dp_cost[best_i][S]
        # backtrack
        path: List[Tuple[str, Optional[int]]] = []
        i, j = best_i, S
        while j > 0:
            decision = dp_dec[i][j]
            path.append(decision)
            i, j = dp_prev[i][j]
        path.reverse()  # decisions per slot 0..S-1
        return path, total_cost, [dp_cost, dp_prev, dp_dec]

    # ---- Consistency checks ----
    def is_format_consistent(self, seq: List[SlotResult]) -> bool:
        if len(seq) != 8:
            return False

        # index 0,1 digits (or unknown "?"), index 2 letter (or "?"), 3..7 digits (or "?")
        def is_digit_or_unknown(ch: str) -> bool:
            return ch == "?" or is_digit(ch)

        def is_letter_or_unknown(ch: str) -> bool:
            return ch == "?" or is_letter(ch)

        return (
                is_digit_or_unknown(seq[0].char) and
                is_digit_or_unknown(seq[1].char) and
                is_letter_or_unknown(seq[2].char) and
                all(is_digit_or_unknown(seq[j].char) for j in range(3, 8))
        )

    # ---- Construct sequence from path ----
    def construct_sequence(
            self,
            path: List[Tuple[str, Optional[int]]],
            dets: List[Detection],
            x_slots: List[float],
            slot_types: List[str],
            det_index_map: Dict[int, int],  # maps det global idx -> original index
    ) -> List[SlotResult]:
        seq: List[SlotResult] = []
        for j, decision in enumerate(path):
            if decision[0] == 'assign' and decision[1] is not None:
                di = decision[1]
                d = dets[di]
                ch = d.char
                if slot_types[j] == 'digit' and not is_digit(ch):
                    ch = "?"
                if slot_types[j] == 'letter' and not is_letter(ch):
                    ch = "?"
                seq.append(SlotResult(
                    index=j,
                    char=ch,
                    conf=float(d.char_conf),
                    assigned_detection=det_index_map.get(di, di),
                    bbox=d.bbox,
                ))
            else:
                seq.append(SlotResult(
                    index=j,
                    char="?",
                    conf=0.0,
                    assigned_detection=None,
                    bbox=None,
                ))
        return seq

    # ---- Edge-drop heuristic ----
    def drop_low_conf_edge(self, dets_sorted: List[int], dets: List[Detection]) -> Optional[int]:
        """Return index of a low-confidence edge detection to drop, or None."""
        if not dets_sorted:
            return None
        left = dets[dets_sorted[0]]
        right = dets[dets_sorted[-1]]
        cand = []
        if left.char_conf < self.th_edge_drop:
            cand.append((left.char_conf, dets_sorted[0]))
        if right.char_conf < self.th_edge_drop:
            cand.append((right.char_conf, dets_sorted[-1]))
        if not cand:
            return None
        cand.sort(key=lambda x: x[0])  # lowest conf first
        return cand[0][1]

    # ---- Public API: align plate ----
    def align(
            self,
            detections: List[Detection],
            rectified: bool = True,
    ) -> List[AlignmentResult]:
        """
        Returns a list of AlignmentResult.
        - If a confident letter anchor exists, returns a single best alignment.
        - If not, returns up to 3 hypotheses with probabilities.
        """

        if not detections:
            # all unknown slots
            empty_seq = [SlotResult(i, "?", 0.0, None, None) for i in range(8)]
            return [AlignmentResult(empty_seq, [i * 10.0 for i in range(8)], math.log(1.0), 0.0, [None] * 8)]

        # Sort detections by x
        idx_sorted = sorted(range(len(detections)), key=lambda i: detections[i].x)
        a = self.estimate_advance(detections)

        # Identify letter candidates
        letter_cands = [i for i in idx_sorted if detections[i].p_letter > self.th_letter]

        # print('==== debug ====')
        # print('letter_conds = ',letter_cands)

        # Helper to build det index map
        det_index_map = {i: i for i in range(len(detections))}

        # ---------------------- Case A: Letter-known mode ---------------------
        if len(letter_cands) > 0:
            # choose the strongest letter as anchor
            anchor = max(letter_cands, key=lambda i: detections[i].p_letter)
            x_slots = self.expected_centers_from_anchor(detections[anchor].x, a)
            slot_types = PATTERN
            path, total_cost, dp = self.align_with_dp(idx_sorted, detections, x_slots, slot_types, a, False)
            seq = self.construct_sequence(path, detections, x_slots, slot_types, det_index_map)

            # print('output seq 1 = ',seq)

            # Consistency check and edge-drop-and-retry
            if not self.is_format_consistent(seq):
                to_drop = self.drop_low_conf_edge(idx_sorted, detections)
                if to_drop is not None:
                    dets2 = [d for k, d in enumerate(detections) if k != to_drop]
                    # rebuild index maps
                    remap = {}
                    orig_to_new = {}
                    new_to_orig = {}
                    j = 0
                    for k in range(len(detections)):
                        if k == to_drop:
                            continue
                        remap[j] = k
                        orig_to_new[k] = j
                        new_to_orig[j] = k
                        j += 1
                    idx_sorted2 = sorted(range(len(dets2)), key=lambda i: dets2[i].x)
                    a2 = self.estimate_advance(dets2)
                    x_slots2 = self.expected_centers_from_anchor(detections[anchor].x, a2)
                    path2, total_cost2, _ = self.align_with_dp(idx_sorted2, dets2, x_slots2, slot_types, a2, False)
                    # Use det_index_map mapping new indices to original
                    det_index_map2 = {i: remap[i] for i in range(len(dets2))}
                    seq2 = self.construct_sequence(path2, dets2, x_slots2, slot_types, det_index_map2)
                    if self.is_format_consistent(seq2):
                        return [AlignmentResult(
                            sequence=seq2,
                            expected_centers=x_slots2,
                            alignment_logprob=-total_cost2,
                            total_cost=total_cost2,
                            used_detections=[r.assigned_detection for r in seq2],
                        )]
                # cannot resolve after drop
                return [AlignmentResult(
                    sequence=seq,
                    expected_centers=x_slots,
                    alignment_logprob=-total_cost,
                    total_cost=total_cost,
                    used_detections=[r.assigned_detection for r in seq],
                )]

            # Normal successful path
            return [AlignmentResult(
                sequence=seq,
                expected_centers=x_slots,
                alignment_logprob=-total_cost,
                total_cost=total_cost,
                used_detections=[r.assigned_detection for r in seq],
            )]

        # -------------------- Case B: Letter-unknown mode ---------------------
        # choose a reference detection (median by x)
        mid = idx_sorted[len(idx_sorted) // 2]
        ref_x = detections[mid].x
        candidates: List[Tuple[float, AlignmentResult]] = []
        slot_types = PATTERN

        for ref_slot in range(8):
            x_slots = self.expected_centers_from_offset(ref_x, a, ref_slot)
            path, total_cost, _ = self.align_with_dp(idx_sorted, detections, x_slots, slot_types, a, True)
            seq = self.construct_sequence(path, detections, x_slots, slot_types, det_index_map)
            candidates.append((total_cost, AlignmentResult(
                sequence=seq,
                expected_centers=x_slots,
                alignment_logprob=-total_cost,
                total_cost=total_cost,
                used_detections=[r.assigned_detection for r in seq],
            )))

        # sort by cost, keep top-K
        candidates.sort(key=lambda t: t[0])
        topK = [c[1] for c in candidates[:3]]
        # Try edge drop once if the best is inconsistent
        if topK:
            if not self.is_format_consistent(topK[0].sequence):
                to_drop = self.drop_low_conf_edge(idx_sorted, detections)
                if to_drop is not None:
                    dets2 = [d for k, d in enumerate(detections) if k != to_drop]
                    idx_sorted2 = sorted(range(len(dets2)), key=lambda i: dets2[i].x)
                    a2 = self.estimate_advance(dets2)
                    candidates2: List[Tuple[float, AlignmentResult]] = []
                    mid2 = idx_sorted2[len(idx_sorted2) // 2]
                    ref_x2 = dets2[mid2].x
                    for ref_slot in range(8):
                        x_slots2 = self.expected_centers_from_offset(ref_x2, a2, ref_slot)
                        path2, total_cost2, _ = self.align_with_dp(idx_sorted2, dets2, x_slots2, slot_types, a2, True)
                        # mapping new indices back to originals
                        remap = {}
                        j = 0
                        for k in range(len(detections)):
                            if k == to_drop:
                                continue
                            remap[j] = k
                            j += 1
                        det_index_map2 = {i: remap[i] for i in range(len(dets2))}
                        seq2 = self.construct_sequence(path2, dets2, x_slots2, slot_types, det_index_map2)
                        candidates2.append((total_cost2, AlignmentResult(
                            sequence=seq2,
                            expected_centers=x_slots2,
                            alignment_logprob=-total_cost2,
                            total_cost=total_cost2,
                            used_detections=[r.assigned_detection for r in seq2],
                        )))
                    candidates2.sort(key=lambda t: t[0])
                    topK = [c[1] for c in candidates2[:3]]

        # Normalize probabilities from logprobs
        logps = [r.alignment_logprob for r in topK]
        probs = softmax(logps)
        for r, p in zip(topK, probs):
            r.alignment_logprob = math.log(max(1e-12, p))
        return topK


def yolo_to_detections(results, model): #todo new func confidence:
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        probs = box.probs.tolist() if hasattr(box, "probs") else None

        if probs:  # YOLO gave multi-class probabilities
            class_probs = {model.names[i]: p for i, p in enumerate(probs)}
            # print(class_probs)
        else:      # fallback: only top-1
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            # print(cls_id,conf)
            class_probs = {model.names[cls_id]: conf}
            # print(class_probs)

        detections.append(Detection(bbox=(x1, y1, x2, y2), class_probs=class_probs))
        # print(Detection(bbox=(x1, y1, x2, y2), class_probs=class_probs))

    # ✅ Sort detections by leftmost x (x1)
    detections.sort(key=lambda d: d.bbox[0])

    return detections


def rotate_plate_and_yolos(img_or_path, yolo_result):
    if isinstance(img_or_path, str):
        image = cv2.imread(img_or_path)
        if image is None:
            raise ValueError(f"Could not read image from {img_or_path}")
    elif isinstance(img_or_path, np.ndarray):
        image = img_or_path
    else:
        raise TypeError("img_or_path must be str path or np.ndarray image")

    yolo_boxes = yolo_result.boxes
    # --- Extract YOLO detections ---
    xyxy = yolo_boxes.xyxy.cpu().numpy()
    conf = yolo_boxes.conf.cpu().numpy()
    cls = yolo_boxes.cls.cpu().numpy()

    if xyxy.shape[0] < 2:
        raise ValueError("Need at least two bounding boxes to fit a line.")

    # --- Compute centers ---
    centers = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2,
                               (xyxy[:, 1] + xyxy[:, 3]) / 2))

    # --- Fit a line to centers ---
    [vx, vy, x0, y0] = cv2.fitLine(centers.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    angle = atan2(vy, vx)

    # --- Rotation matrix ---
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_deg = degrees(angle)
    M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)

    # --- Rotate image ---
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    # --- Rotate bounding boxes ---
    rotated_xyxy = []
    for box in xyxy:
        x1, y1, x2, y2 = box
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_hom = np.hstack([corners, ones])
        rotated_corners = M.dot(corners_hom.T).T
        new_x1, new_y1 = np.min(rotated_corners[:, 0]), np.min(rotated_corners[:, 1])
        new_x2, new_y2 = np.max(rotated_corners[:, 0]), np.max(rotated_corners[:, 1])
        rotated_xyxy.append([new_x1, new_y1, new_x2, new_y2])

    rotated_xyxy = torch.tensor(rotated_xyxy, dtype=torch.float32)
    conf = torch.tensor(conf, dtype=torch.float32).unsqueeze(1)
    cls = torch.tensor(cls, dtype=torch.float32).unsqueeze(1)

    # --- Rebuild box tensor ---
    data = torch.cat([rotated_xyxy, conf, cls], dim=1)  # [N,6]

    # --- Build a full Results object (let Results handle Boxes creation) ---
    rotated_results = Results(
        orig_img=rotated_image,
        path=getattr(yolo_result, "path", None),
        names=yolo_result.names,
        boxes=data  # ✅ pass raw tensor, not Boxes
    )
    # print(rotated_results.names)

    return rotated_image, rotated_results


def car_post_process(yolo_result,car_ocr_model,shape):
    dets2 = yolo_to_detections(yolo_result, car_ocr_model)
    aligner = PlateAligner(alpha=1.15)
    results = aligner.align(dets2, rectified=True)

    valid_results = [res for res in results if all(0 <= cx < shape[1] for cx in res.expected_centers)]
    # If at least one valid hypothesis exists, keep only those
    if valid_results:
        results = valid_results

    print('All hypothesis: ')
    # Print outputs
    for k, res in enumerate(results):
        pred_text2 = ''.join([r.char for r in res.sequence])
        print(pred_text2)
    print('='*10)
    return ''.join([r.char for r in results[0].sequence])


def get_yolo_ocr_result_car(results, conf_thresh=0.6):
    boxes = results[0].boxes
    names = results[0].names

    preds = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        coords = [int(x1), int(y1), int(x2), int(y2)]

        # Confidence-based display
        label_to_show = label if conf >= conf_thresh else "?"

        preds.append((x1, label_to_show))  # Save x1 for sorting
    
    # Sort by x1 (left to right)
    preds.sort(key=lambda x: x[0])

    # Return only the labels
    return "".join([label for _, label in preds])

