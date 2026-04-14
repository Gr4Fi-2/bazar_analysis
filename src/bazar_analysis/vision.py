from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageDraw
from rapidfuzz import fuzz

from .utils import normalize_name


@dataclass
class ReferenceFeature:
    entity_id: str
    name: str
    normalized_name: str
    image_path: Path
    phash: imagehash.ImageHash
    mean_rgb: tuple[float, float, float]
    hsv_hist: np.ndarray
    orb_descriptors: np.ndarray | None


@dataclass
class CropBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class MatchCandidate:
    entity_id: str
    name: str
    confidence: float
    detail: dict


@dataclass
class RankBadgeFeature:
    screenshot_id: int
    tier: str
    phash: imagehash.ImageHash
    mean_rgb: tuple[float, float, float]
    hue_histogram: np.ndarray
    saturation_mean: float
    value_mean: float


@dataclass
class RankFeatureSample:
    rank_tier: str
    phash: imagehash.ImageHash
    mean_rgb: tuple[float, float, float]
    hsv_hist: np.ndarray


def load_reference_features(rows) -> list[ReferenceFeature]:
    orb = cv2.ORB_create(nfeatures=128)
    features: list[ReferenceFeature] = []
    for row in rows:
        image_path = row["image_path"]
        if not image_path or not Path(image_path).exists():
            continue
        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize((96, 96))
        array = np.array(resized)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        _, descriptors = orb.detectAndCompute(gray, None)
        features.append(
            ReferenceFeature(
                entity_id=row["entity_id"],
                name=row["name"],
                normalized_name=row["normalized_name"],
                image_path=Path(image_path),
                phash=imagehash.phash(resized),
                mean_rgb=tuple(float(v) for v in array.mean(axis=(0, 1))),
                hsv_hist=_compute_hsv_histogram(array),
                orb_descriptors=descriptors,
            )
        )
    return features


def _compute_hsv_histogram(rgb_array: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def _image_signature(image: Image.Image, size: int = 96) -> tuple[Image.Image, np.ndarray, tuple[float, float, float], np.ndarray]:
    resized = image.convert("RGB").resize((size, size))
    array = np.array(resized)
    mean_rgb = tuple(float(v) for v in array.mean(axis=(0, 1)))
    return resized, array, mean_rgb, _compute_hsv_histogram(array)


def save_crop(image: Image.Image, crop_box: CropBox, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((crop_box.x, crop_box.y, crop_box.x + crop_box.w, crop_box.y + crop_box.h)).save(output_path)


def relative_box(width: int, height: int, x0: float, y0: float, x1: float, y1: float) -> CropBox:
    return CropBox(int(width * x0), int(height * y0), int(width * (x1 - x0)), int(height * (y1 - y0)))


def default_regions(width: int, height: int) -> dict[str, CropBox]:
    return {
        "rank": relative_box(width, height, 0.0, 0.0, 0.23, 0.20),
        "board": relative_box(width, height, 0.38, 0.30, 0.95, 0.58),
        "skills": relative_box(width, height, 0.38, 0.58, 0.88, 0.73),
    }


def detect_slot_boxes(board_image: np.ndarray, origin_x: int, origin_y: int) -> list[CropBox]:
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[CropBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 4500 or area > 140000:
            continue
        aspect = w / max(h, 1)
        if not 0.75 <= aspect <= 1.3:
            continue
        if w < board_image.shape[1] * 0.07 or w > board_image.shape[1] * 0.35:
            continue
        boxes.append(CropBox(origin_x + x, origin_y + y, w, h))

    deduped: list[CropBox] = []
    for box in sorted(boxes, key=lambda item: (item.y, item.x)):
        if any(abs(box.x - prev.x) < 12 and abs(box.y - prev.y) < 12 for prev in deduped):
            continue
        deduped.append(box)
    if len(deduped) >= 5:
        return deduped[:12]
    return []


def fallback_grid(board_box: CropBox) -> list[CropBox]:
    boxes: list[CropBox] = []
    cols = 6
    rows = 1
    cell_w = board_box.w / cols
    cell_h = board_box.h / rows
    margin_x = int(cell_w * 0.04)
    margin_y = int(cell_h * 0.10)
    for row in range(rows):
        for col in range(cols):
            x = int(board_box.x + col * cell_w + margin_x)
            y = int(board_box.y + row * cell_h + margin_y)
            w = int(cell_w - 2 * margin_x)
            h = int(cell_h - 2 * margin_y)
            boxes.append(CropBox(x, y, w, h))
    return boxes


def fallback_skill_grid(skill_box: CropBox) -> list[CropBox]:
    boxes: list[CropBox] = []
    cols = 9
    cell_w = skill_box.w / cols
    size = int(min(cell_w * 0.82, skill_box.h * 0.92))
    y = skill_box.y + int((skill_box.h - size) * 0.48)
    for col in range(cols):
        x = int(skill_box.x + col * cell_w + (cell_w - size) / 2)
        boxes.append(CropBox(x, y, size, size))
    return boxes


def inset_box(box: CropBox, left: float, top: float, right: float, bottom: float) -> CropBox:
    x = int(box.x + box.w * left)
    y = int(box.y + box.h * top)
    w = int(box.w * (right - left))
    h = int(box.h * (bottom - top))
    return CropBox(x, y, max(1, w), max(1, h))


def square_box(box: CropBox, anchor_x: float = 0.5, anchor_y: float = 0.5) -> CropBox:
    size = max(1, min(box.w, box.h))
    max_dx = max(0, box.w - size)
    max_dy = max(0, box.h - size)
    x = box.x + int(max_dx * min(max(anchor_x, 0.0), 1.0))
    y = box.y + int(max_dy * min(max(anchor_y, 0.0), 1.0))
    return CropBox(x, y, size, size)


def item_focus_boxes(slot_box: CropBox) -> list[tuple[str, CropBox]]:
    art_box = inset_box(slot_box, 0.05, 0.06, 0.73, 0.88)
    return [
        ("card", art_box),
        ("left_square", square_box(art_box, anchor_x=0.0, anchor_y=0.5)),
        ("center_square", square_box(inset_box(slot_box, 0.10, 0.10, 0.80, 0.84), anchor_x=0.5, anchor_y=0.5)),
    ]


def item_crop_variants(box: CropBox) -> list[tuple[str, CropBox]]:
    return [
        ("card", inset_box(box, 0.12, 0.06, 0.88, 0.82)),
        ("icon", inset_box(box, 0.24, 0.14, 0.76, 0.66)),
        ("icon_tight", inset_box(box, 0.29, 0.18, 0.71, 0.60)),
    ]


def rank_badge_variants(box: CropBox) -> list[tuple[str, CropBox]]:
    return [
        ("badge", inset_box(box, 0.02, 0.08, 0.30, 0.78)),
        ("badge_tight", inset_box(box, 0.05, 0.14, 0.24, 0.66)),
    ]


def _candidate_score(crop_image: Image.Image, crop_array: np.ndarray, reference: ReferenceFeature, name_hint: str | None = None) -> MatchCandidate:
    orb = cv2.ORB_create(nfeatures=128)
    resized, resized_array, mean_rgb, hsv_hist = _image_signature(crop_image)
    phash_distance = crop_hash_distance(resized, reference.phash)
    color_distance = math.sqrt(sum((mean_rgb[idx] - reference.mean_rgb[idx]) ** 2 for idx in range(3))) / 441.67295593
    hist_score = hsv_hist_similarity(hsv_hist, reference.hsv_hist)
    gray = cv2.cvtColor(resized_array, cv2.COLOR_RGB2GRAY)
    _, descriptors = orb.detectAndCompute(gray, None)
    orb_score = orb_similarity(descriptors, reference.orb_descriptors)
    hint_score = 0.0
    if name_hint:
        hint_score = fuzz.partial_ratio(normalize_name(name_hint), reference.normalized_name) / 100.0

    score = max(0.0, 1.0 - (phash_distance / 20.0)) * 0.30
    score += max(0.0, 1.0 - color_distance) * 0.10
    score += hist_score * 0.25
    score += orb_score * 0.25
    score += hint_score * 0.10
    return MatchCandidate(
        entity_id=reference.entity_id,
        name=reference.name,
        confidence=round(min(score, 0.999), 4),
        detail={
            "phash_distance": phash_distance,
            "color_distance": round(color_distance, 4),
            "hist_score": round(hist_score, 4),
            "orb_score": round(orb_score, 4),
            "hint_score": round(hint_score, 4),
        },
    )


def crop_hash_distance(image: Image.Image, reference_hash: imagehash.ImageHash) -> int:
    return int(imagehash.phash(image) - reference_hash)


def orb_similarity(descriptors_a: np.ndarray | None, descriptors_b: np.ndarray | None) -> float:
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda item: item.distance)
    good = [match for match in matches[:20] if match.distance < 50]
    return min(1.0, len(good) / 12.0)


def hsv_hist_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    score = cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def match_crop(crop_image: Image.Image, references: list[ReferenceFeature], name_hints: list[str] | None = None, top_n: int = 5) -> list[MatchCandidate]:
    hints = name_hints or []
    crop_array = np.array(crop_image.convert("RGB"))
    candidates: list[MatchCandidate] = []
    for reference in references:
        hint = max(hints, key=lambda item: fuzz.partial_ratio(normalize_name(item), reference.normalized_name), default=None)
        candidates.append(_candidate_score(crop_image, crop_array, reference, name_hint=hint))
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates[:top_n]


def aggregate_match_candidates(candidate_groups: list[tuple[str, list[MatchCandidate]]], top_n: int = 5) -> list[MatchCandidate]:
    merged: dict[str, dict] = {}
    for variant_name, candidates in candidate_groups:
        for rank_index, candidate in enumerate(candidates):
            entry = merged.setdefault(
                candidate.entity_id,
                {
                    "name": candidate.name,
                    "best_confidence": 0.0,
                    "best_detail": {},
                    "best_variant": variant_name,
                    "support": 0,
                    "top_hits": 0,
                    "confidence_sum": 0.0,
                },
            )
            entry["support"] += 1
            entry["confidence_sum"] += candidate.confidence
            if rank_index == 0:
                entry["top_hits"] += 1
            if candidate.confidence > entry["best_confidence"]:
                entry["best_confidence"] = candidate.confidence
                entry["best_detail"] = candidate.detail
                entry["best_variant"] = variant_name

    aggregated: list[MatchCandidate] = []
    for entity_id, entry in merged.items():
        avg_confidence = entry["confidence_sum"] / max(entry["support"], 1)
        support_bonus = 0.035 * max(0, entry["support"] - 1)
        top_hit_bonus = 0.025 * max(0, entry["top_hits"] - 1)
        confidence = min(0.999, entry["best_confidence"] + support_bonus + top_hit_bonus)
        detail = dict(entry["best_detail"])
        detail.update(
            {
                "avg_confidence": round(avg_confidence, 4),
                "best_variant": entry["best_variant"],
                "support": entry["support"],
                "top_hits": entry["top_hits"],
            }
        )
        aggregated.append(
            MatchCandidate(
                entity_id=entity_id,
                name=entry["name"],
                confidence=round(confidence, 4),
                detail=detail,
            )
        )
    aggregated.sort(key=lambda item: (item.confidence, item.detail.get("support", 0), item.detail.get("top_hits", 0)), reverse=True)
    return aggregated[:top_n]


def extract_rank_badge(rank_crop: Image.Image) -> Image.Image:
    width, height = rank_crop.size
    badge_box = (int(width * 0.02), int(height * 0.02), int(width * 0.34), int(height * 0.92))
    return rank_crop.crop(badge_box)


def _hue_histogram(hsv_array: np.ndarray) -> np.ndarray:
    saturation = hsv_array[:, :, 1]
    value = hsv_array[:, :, 2]
    mask = (saturation >= 40) & (value >= 40)
    hues = hsv_array[:, :, 0][mask]
    if hues.size == 0:
        hues = hsv_array[:, :, 0].reshape(-1)
    histogram, _ = np.histogram(hues, bins=18, range=(0, 180), density=False)
    histogram = histogram.astype(np.float32)
    total = float(histogram.sum())
    return histogram / total if total else histogram


def build_rank_badge_feature(rank_crop: Image.Image, tier: str, screenshot_id: int) -> RankBadgeFeature:
    badge = extract_rank_badge(rank_crop).convert("RGB").resize((72, 72))
    array = np.array(badge)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    return RankBadgeFeature(
        screenshot_id=screenshot_id,
        tier=tier,
        phash=imagehash.phash(badge),
        mean_rgb=tuple(float(v) for v in array.mean(axis=(0, 1))),
        hue_histogram=_hue_histogram(hsv),
        saturation_mean=float(hsv[:, :, 1].mean() / 255.0),
        value_mean=float(hsv[:, :, 2].mean() / 255.0),
    )


def match_rank_badge(rank_crop: Image.Image, references: list[RankBadgeFeature], exclude_screenshot_id: int | None = None, top_n: int = 5) -> list[dict]:
    if not references:
        return []

    feature = build_rank_badge_feature(rank_crop, tier="Unknown", screenshot_id=exclude_screenshot_id or -1)
    grouped_scores: dict[str, list[dict]] = defaultdict(list)
    for reference in references:
        if exclude_screenshot_id is not None and reference.screenshot_id == exclude_screenshot_id:
            continue
        phash_score = max(0.0, 1.0 - (int(feature.phash - reference.phash) / 18.0))
        rgb_distance = math.sqrt(sum((feature.mean_rgb[idx] - reference.mean_rgb[idx]) ** 2 for idx in range(3))) / 441.67295593
        rgb_score = max(0.0, 1.0 - rgb_distance)
        hue_score = float(np.minimum(feature.hue_histogram, reference.hue_histogram).sum())
        sat_score = max(0.0, 1.0 - abs(feature.saturation_mean - reference.saturation_mean))
        value_score = max(0.0, 1.0 - abs(feature.value_mean - reference.value_mean))
        confidence = min(0.999, phash_score * 0.38 + rgb_score * 0.20 + hue_score * 0.28 + sat_score * 0.08 + value_score * 0.06)
        grouped_scores[reference.tier].append(
            {
                "confidence": round(confidence, 4),
                "detail": {
                    "phash_score": round(phash_score, 4),
                    "rgb_score": round(rgb_score, 4),
                    "hue_score": round(hue_score, 4),
                    "sat_score": round(sat_score, 4),
                    "value_score": round(value_score, 4),
                },
            }
        )

    candidates: list[dict] = []
    for tier, scores in grouped_scores.items():
        scores.sort(key=lambda item: item["confidence"], reverse=True)
        top_scores = scores[:3]
        confidence = sum(item["confidence"] for item in top_scores) / len(top_scores)
        candidates.append(
            {
                "rank": tier,
                "confidence": round(confidence, 4),
                "detail": {
                    "prototype_count": len(scores),
                    "top_match": top_scores[0],
                    "top3_mean": round(confidence, 4),
                },
            }
        )
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates[:top_n]


def build_rank_feature_sample(rank_tier: str, crop_image: Image.Image) -> RankFeatureSample:
    resized, _array, mean_rgb, hsv_hist = _image_signature(crop_image, size=72)
    return RankFeatureSample(
        rank_tier=rank_tier,
        phash=imagehash.phash(resized),
        mean_rgb=mean_rgb,
        hsv_hist=hsv_hist,
    )


def match_rank_crop(crop_image: Image.Image, samples: list[RankFeatureSample], title_hint: str | None = None, top_n: int = 5) -> list[MatchCandidate]:
    if not samples:
        return []

    resized, _array, mean_rgb, hsv_hist = _image_signature(crop_image, size=72)
    grouped: dict[str, list[float]] = {}
    details: dict[str, dict[str, float | int]] = {}
    for sample in samples:
        phash_distance = crop_hash_distance(resized, sample.phash)
        color_distance = math.sqrt(sum((mean_rgb[idx] - sample.mean_rgb[idx]) ** 2 for idx in range(3))) / 441.67295593
        hist_score = hsv_hist_similarity(hsv_hist, sample.hsv_hist)
        score = max(0.0, 1.0 - (phash_distance / 18.0)) * 0.35
        score += max(0.0, 1.0 - color_distance) * 0.15
        score += hist_score * 0.50
        grouped.setdefault(sample.rank_tier, []).append(score)

    candidates: list[MatchCandidate] = []
    normalized_hint = normalize_name(title_hint) if title_hint else None
    for rank_tier, scores in grouped.items():
        ordered = sorted(scores, reverse=True)
        best_score = ordered[0]
        avg_score = sum(ordered[: min(4, len(ordered))]) / min(4, len(ordered))
        hint_bonus = 0.06 if normalized_hint and normalize_name(rank_tier) == normalized_hint else 0.0
        confidence = min(0.999, best_score * 0.75 + avg_score * 0.20 + hint_bonus)
        candidates.append(
            MatchCandidate(
                entity_id=rank_tier,
                name=rank_tier,
                confidence=round(confidence, 4),
                detail={
                    "best_sample_score": round(best_score, 4),
                    "avg_top_score": round(avg_score, 4),
                    "sample_count": len(scores),
                    "hint_bonus": round(hint_bonus, 4),
                },
            )
        )

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates[:top_n]


def annotate_image(image: Image.Image, annotations: list[tuple[CropBox, str, str]], output_path: Path) -> None:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for box, label, color in annotations:
        draw.rectangle((box.x, box.y, box.x + box.w, box.y + box.h), outline=color, width=3)
        draw.text((box.x + 3, max(0, box.y - 14)), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def candidate_payload(candidates: list[MatchCandidate]) -> str:
    return json.dumps(
        [
            {
                "entity_id": candidate.entity_id,
                "name": candidate.name,
                "confidence": candidate.confidence,
                "detail": candidate.detail,
            }
            for candidate in candidates
        ],
        ensure_ascii=True,
        sort_keys=True,
    )
