from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .db import next_id
from .utils import normalize_name
from .vision import (
    CropBox,
    annotate_image,
    build_rank_feature_sample,
    candidate_payload,
    default_regions,
    fallback_grid,
    fallback_skill_grid,
    inset_box,
    item_crop_variants,
    load_reference_features,
    match_rank_crop,
    match_crop,
    rank_badge_variants,
    save_crop,
)


def _load_reference_sets(conn):
    item_rows = conn.execute("SELECT * FROM reference_items WHERE image_path IS NOT NULL ORDER BY name").fetchall()
    skill_rows = conn.execute("SELECT * FROM reference_skills WHERE image_path IS NOT NULL ORDER BY name").fetchall()
    item_features = load_reference_features(item_rows)
    skill_features = load_reference_features(skill_rows)
    return item_features, skill_features


def _hint_matched_item_features(item_features, item_hints: list[str]):
    if not item_hints:
        return item_features
    normalized_hints = {normalize_name(hint) for hint in item_hints if hint.strip()}
    matched = [feature for feature in item_features if feature.normalized_name in normalized_hints]
    return matched or item_features


def _queue_review(conn, screenshot_id: int, detection_type: str, crop_path: str, confidence: float, raw_label: str | None, top_candidates_json: str) -> None:
    existing_review = conn.execute(
        "SELECT review_id FROM review_queue WHERE screenshot_id = ? AND detection_type = ? AND crop_path = ?",
        (screenshot_id, detection_type, crop_path),
    ).fetchone()
    if existing_review:
        conn.execute(
            """
            UPDATE review_queue
            SET confidence = ?, raw_label = ?, top_candidates_json = ?, status = 'pending'
            WHERE review_id = ?
            """,
            (confidence, raw_label, top_candidates_json, existing_review["review_id"]),
        )
        return
    conn.execute(
        """
        INSERT INTO review_queue(review_id, screenshot_id, detection_type, crop_path, confidence, raw_label, top_candidates_json)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (next_id(conn, "review_queue", "review_id"), screenshot_id, detection_type, crop_path, confidence, raw_label, top_candidates_json),
    )


def _build_rank_reference_samples(screenshots) -> list:
    rank_samples = []
    for screenshot in screenshots:
        rank_tier = screenshot["rank_title_hint"]
        image_path = screenshot["local_path"]
        if not rank_tier or not image_path:
            continue
        path = Path(image_path)
        if not path.exists():
            continue
        try:
            with Image.open(path) as raw_image:
                image = raw_image.convert("RGB")
                rank_box = default_regions(*image.size)["rank"]
                for _variant_name, badge_box in rank_badge_variants(rank_box):
                    crop = image.crop((badge_box.x, badge_box.y, badge_box.x + badge_box.w, badge_box.y + badge_box.h))
                    rank_samples.append(build_rank_feature_sample(rank_tier, crop))
        except Exception:
            continue
    return rank_samples


def _match_item_slot(image: Image.Image, box: CropBox, item_features, item_hints: list[str]):
    aggregated: dict[str, dict] = {}
    variant_results: list[tuple[str, CropBox, list]] = []
    for variant_name, variant_box in item_crop_variants(box):
        crop = image.crop((variant_box.x, variant_box.y, variant_box.x + variant_box.w, variant_box.y + variant_box.h))
        candidates = match_crop(crop, item_features, name_hints=item_hints)
        variant_results.append((variant_name, variant_box, candidates))
        for candidate in candidates:
            current = aggregated.get(candidate.entity_id)
            if current is None:
                aggregated[candidate.entity_id] = {
                    "name": candidate.name,
                    "scores": [candidate.confidence],
                    "details": [dict(candidate.detail, variant=variant_name)],
                    "best_box": variant_box,
                    "best_variant": variant_name,
                    "best_confidence": candidate.confidence,
                }
                continue
            current["scores"].append(candidate.confidence)
            current["details"].append(dict(candidate.detail, variant=variant_name))
            if candidate.confidence > current["best_confidence"]:
                current["best_box"] = variant_box
                current["best_variant"] = variant_name
                current["best_confidence"] = candidate.confidence

    merged_candidates = []
    for entity_id, payload in aggregated.items():
        scores = sorted(payload["scores"], reverse=True)
        best_score = scores[0]
        avg_score = sum(scores) / len(scores)
        agreement_bonus = min(0.08, 0.04 * (len(scores) - 1))
        merged_candidates.append(
            {
                "entity_id": entity_id,
                "name": payload["name"],
                "confidence": round(min(0.999, best_score * 0.82 + avg_score * 0.12 + agreement_bonus), 4),
                "detail": {
                    "best_variant": payload["best_variant"],
                    "variant_hits": len(scores),
                    "best_raw_confidence": round(best_score, 4),
                    "avg_raw_confidence": round(avg_score, 4),
                    "agreement_bonus": round(agreement_bonus, 4),
                    "variant_details": payload["details"],
                },
                "box": payload["best_box"],
            }
        )

    merged_candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return merged_candidates[:5], variant_results


def extract_board_data(conn, settings: Settings) -> dict[str, int]:
    item_features, skill_features = _load_reference_sets(conn)
    conn.execute(
        "DELETE FROM extracted_board_items WHERE screenshot_id IN (SELECT screenshot_id FROM screenshots WHERE is_primary = 1)"
    )
    conn.execute(
        "DELETE FROM extracted_skills WHERE screenshot_id IN (SELECT screenshot_id FROM screenshots WHERE is_primary = 1)"
    )
    conn.execute(
        "DELETE FROM extracted_ranks WHERE screenshot_id IN (SELECT screenshot_id FROM screenshots WHERE is_primary = 1)"
    )
    conn.execute(
        "DELETE FROM review_queue WHERE screenshot_id IN (SELECT screenshot_id FROM screenshots WHERE is_primary = 1)"
    )
    screenshots = conn.execute(
        """
        SELECT s.*, p.title, p.rank_title_hint, p.item_hints_json, p.post_url
        FROM screenshots s
        JOIN posts p ON p.post_id = s.post_id
        WHERE s.local_path IS NOT NULL AND s.is_primary = 0
        ORDER BY s.screenshot_id
        """
    ).fetchall()
    print(f"[extract] processing {len(screenshots)} board screenshots", flush=True)
    rank_samples = _build_rank_reference_samples(screenshots)
    print(f"[extract] built {len(rank_samples)} rank samples", flush=True)

    processed = 0
    item_detections = 0
    skill_detections = 0
    rank_detections = 0

    for index, screenshot in enumerate(screenshots, start=1):
        screenshot_id = screenshot["screenshot_id"]
        if index == 1 or index % 10 == 0 or index == len(screenshots):
            print(
                f"[extract] screenshot {index}/{len(screenshots)} id={screenshot_id} items={item_detections} skills={skill_detections} ranks={rank_detections}",
                flush=True,
            )
        image_path = Path(screenshot["local_path"])
        if not image_path.exists():
            continue
        if (screenshot["width"] or 0) < 1000 or (screenshot["height"] or 0) < 600:
            conn.execute("DELETE FROM review_queue WHERE screenshot_id = ?", (screenshot_id,))
            _queue_review(
                conn,
                screenshot_id,
                "screenshot_layout",
                str(image_path),
                0.0,
                "small_or_non_board_image",
                json.dumps(
                    {
                        "width": screenshot["width"],
                        "height": screenshot["height"],
                        "local_path": str(image_path),
                        "post_url": screenshot["post_url"],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            )
            processed += 1
            continue
        item_hints = json.loads(screenshot["item_hints_json"])
        matched_item_features = _hint_matched_item_features(item_features, item_hints)
        item_confidence_threshold = 0.30 if matched_item_features is not item_features else 0.38

        conn.execute("DELETE FROM extracted_board_items WHERE screenshot_id = ?", (screenshot_id,))
        conn.execute("DELETE FROM extracted_skills WHERE screenshot_id = ?", (screenshot_id,))
        conn.execute("DELETE FROM extracted_ranks WHERE screenshot_id = ?", (screenshot_id,))
        conn.execute("DELETE FROM review_queue WHERE screenshot_id = ?", (screenshot_id,))

        try:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
        except Exception as exc:
            _queue_review(
                conn,
                screenshot_id,
                "screenshot_file",
                str(image_path),
                0.0,
                type(exc).__name__,
                json.dumps(
                    {
                        "error": str(exc),
                        "local_path": str(image_path),
                        "post_url": screenshot["post_url"],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            )
            processed += 1
            continue

        try:
            width, height = image.size
            regions = default_regions(width, height)
            save_crop(image, regions["board"], settings.debug_board_dir / f"board_{screenshot_id}.png")
            save_crop(image, regions["skills"], settings.debug_skill_dir / f"skills_{screenshot_id}.png")
            save_crop(image, regions["rank"], settings.debug_rank_dir / f"rank_{screenshot_id}.png")

            board_boxes = fallback_grid(regions["board"])
            skill_boxes = fallback_skill_grid(regions["skills"])
            annotations: list[tuple[CropBox, str, str]] = []

            predicted_items: list[str] = []
            for slot_index, box in enumerate(board_boxes):
                candidates, variant_results = _match_item_slot(image, box, matched_item_features, item_hints)
                top_candidate = candidates[0] if candidates else None
                focus_box = top_candidate["box"] if top_candidate else inset_box(box, 0.14, 0.08, 0.86, 0.78)
                crop = image.crop((focus_box.x, focus_box.y, focus_box.x + focus_box.w, focus_box.y + focus_box.h))
                crop_path = settings.debug_crops_dir / f"item_{screenshot_id}_{slot_index}.png"
                crop.save(crop_path)
                confidence = top_candidate["confidence"] if top_candidate else 0.0
                entity_id = top_candidate["entity_id"] if top_candidate and confidence >= item_confidence_threshold else None
                raw_label = top_candidate["name"] if top_candidate else None
                status = "ok" if entity_id else "review"
                payload = json.dumps(
                    [
                        {
                            "entity_id": candidate["entity_id"],
                            "name": candidate["name"],
                            "confidence": candidate["confidence"],
                            "detail": candidate["detail"],
                        }
                        for candidate in candidates
                    ]
                    + [
                        {
                            "variant": variant_name,
                            "crop_box": {"x": variant_box.x, "y": variant_box.y, "w": variant_box.w, "h": variant_box.h},
                            "top_candidates": json.loads(candidate_payload(variant_candidates)),
                        }
                        for variant_name, variant_box, variant_candidates in variant_results
                    ],
                    ensure_ascii=True,
                    sort_keys=True,
                )
                conn.execute(
                    """
                    INSERT INTO extracted_board_items(detection_id, screenshot_id, slot_index, entity_id, raw_label, confidence, method, bbox_x, bbox_y, bbox_w, bbox_h, duplicate_count, crop_path, top_candidates_json, status)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        next_id(conn, "extracted_board_items", "detection_id"),
                        screenshot_id,
                        slot_index,
                        entity_id,
                        raw_label,
                        confidence,
                        "icon_match+slot_detection",
                        focus_box.x,
                        focus_box.y,
                        focus_box.w,
                        focus_box.h,
                        None,
                        str(crop_path),
                        payload,
                        status,
                    ),
                )
                annotations.append((focus_box, raw_label or "unknown", "lime" if entity_id else "orange"))
                if entity_id:
                    predicted_items.append(entity_id)
                    item_detections += 1
                else:
                    _queue_review(conn, screenshot_id, "board_item", str(crop_path), confidence, raw_label, payload)

            counts = Counter(predicted_items)
            for entity_id, duplicate_count in counts.items():
                conn.execute(
                    "UPDATE extracted_board_items SET duplicate_count = ? WHERE screenshot_id = ? AND entity_id = ?",
                    (duplicate_count, screenshot_id, entity_id),
                )

            for slot_index, box in enumerate(skill_boxes):
                crop = image.crop((box.x, box.y, box.x + box.w, box.y + box.h))
                crop_path = settings.debug_crops_dir / f"skill_{screenshot_id}_{slot_index}.png"
                crop.save(crop_path)
                candidates = match_crop(crop, skill_features)
                top_candidate = candidates[0] if candidates else None
                confidence = top_candidate.confidence if top_candidate else 0.0
                entity_id = top_candidate.entity_id if top_candidate and confidence >= 0.33 else None
                raw_label = top_candidate.name if top_candidate else None
                status = "ok" if entity_id else "review"
                payload = candidate_payload(candidates)
                conn.execute(
                    """
                    INSERT INTO extracted_skills(detection_id, screenshot_id, slot_index, entity_id, raw_label, confidence, method, bbox_x, bbox_y, bbox_w, bbox_h, crop_path, top_candidates_json, status)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        next_id(conn, "extracted_skills", "detection_id"),
                        screenshot_id,
                        slot_index,
                        entity_id,
                        raw_label,
                        confidence,
                        "icon_match+fixed_skill_grid",
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        str(crop_path),
                        payload,
                        status,
                    ),
                )
                if entity_id:
                    annotations.append((box, raw_label or "skill", "cyan"))
                    skill_detections += 1
                elif confidence >= 0.20:
                    annotations.append((box, raw_label or "skill?", "yellow"))
                    _queue_review(conn, screenshot_id, "skill", str(crop_path), confidence, raw_label, payload)

            rank_box = regions["rank"]
            rank_candidates = []
            rank_variant_results = []
            for variant_name, badge_box in rank_badge_variants(rank_box):
                badge_crop = image.crop((badge_box.x, badge_box.y, badge_box.x + badge_box.w, badge_box.y + badge_box.h))
                candidates = match_rank_crop(badge_crop, rank_samples, title_hint=screenshot["rank_title_hint"])
                rank_variant_results.append((variant_name, badge_box, candidates))
                for candidate in candidates:
                    rank_candidates.append((variant_name, badge_box, candidate))

            aggregated_ranks: dict[str, dict] = {}
            for variant_name, badge_box, candidate in rank_candidates:
                current = aggregated_ranks.get(candidate.entity_id)
                if current is None:
                    aggregated_ranks[candidate.entity_id] = {
                        "scores": [candidate.confidence],
                        "detail": [dict(candidate.detail, variant=variant_name)],
                        "box": badge_box,
                        "best_score": candidate.confidence,
                    }
                    continue
                current["scores"].append(candidate.confidence)
                current["detail"].append(dict(candidate.detail, variant=variant_name))
                if candidate.confidence > current["best_score"]:
                    current["best_score"] = candidate.confidence
                    current["box"] = badge_box

            merged_rank_candidates = []
            for rank_tier, payload in aggregated_ranks.items():
                scores = sorted(payload["scores"], reverse=True)
                best_score = scores[0]
                avg_score = sum(scores) / len(scores)
                agreement_bonus = min(0.06, 0.03 * (len(scores) - 1))
                merged_rank_candidates.append(
                    {
                        "rank_tier": rank_tier,
                        "confidence": round(min(0.999, best_score * 0.82 + avg_score * 0.12 + agreement_bonus), 4),
                        "detail": payload["detail"],
                        "box": payload["box"],
                    }
                )
            merged_rank_candidates.sort(key=lambda item: item["confidence"], reverse=True)

            top_rank = merged_rank_candidates[0] if merged_rank_candidates else None
            rank_focus_box = top_rank["box"] if top_rank else rank_badge_variants(rank_box)[0][1]
            rank_crop_path = settings.debug_crops_dir / f"rank_{screenshot_id}.png"
            image.crop((rank_focus_box.x, rank_focus_box.y, rank_focus_box.x + rank_focus_box.w, rank_focus_box.y + rank_focus_box.h)).save(rank_crop_path)
            hinted_rank = screenshot["rank_title_hint"]
            rank_label = top_rank["rank_tier"] if top_rank and top_rank["confidence"] >= 0.42 else hinted_rank
            rank_confidence = top_rank["confidence"] if top_rank else (0.30 if hinted_rank else 0.05)
            rank_status = "ok" if rank_label else "review"
            rank_payload = json.dumps(
                [
                    {
                        "source": "image_classifier",
                        "rank": candidate["rank_tier"],
                        "confidence": candidate["confidence"],
                        "detail": candidate["detail"],
                    }
                    for candidate in merged_rank_candidates[:5]
                ]
                + [
                    {
                        "source": "variant_candidates",
                        "variant": variant_name,
                        "crop_box": {"x": badge_box.x, "y": badge_box.y, "w": badge_box.w, "h": badge_box.h},
                        "top_candidates": json.loads(candidate_payload(candidates)),
                    }
                    for variant_name, badge_box, candidates in rank_variant_results
                ]
                + [
                    {"source": "title_hint", "rank": hinted_rank, "confidence": 0.30 if hinted_rank else 0.0},
                    {"source": "screenshot_crop_saved", "path": str(rank_crop_path)},
                    {"source": "rank_reference_samples", "count": len(rank_samples)},
                ],
                ensure_ascii=True,
                sort_keys=True,
            )
            conn.execute(
                """
                INSERT INTO extracted_ranks(screenshot_id, raw_label, rank_tier, confidence, method, bbox_x, bbox_y, bbox_w, bbox_h, crop_path, top_candidates_json, status)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    screenshot_id,
                    rank_label,
                    rank_label,
                    rank_confidence,
                    "rank_badge_classifier+title_hint_fallback",
                    rank_focus_box.x,
                    rank_focus_box.y,
                    rank_focus_box.w,
                    rank_focus_box.h,
                    str(rank_crop_path),
                    rank_payload,
                    rank_status,
                ),
            )
            if rank_label:
                annotations.append((rank_focus_box, rank_label, "magenta"))
                rank_detections += 1
            else:
                _queue_review(conn, screenshot_id, "rank", str(rank_crop_path), rank_confidence, None, rank_payload)

            annotate_image(image, annotations, settings.debug_annotated_dir / f"annotated_{screenshot_id}.png")
        finally:
            image.close()
        processed += 1

    conn.commit()
    print(
        f"[extract] done: screenshots={processed}, item_detections={item_detections}, skill_detections={skill_detections}, rank_detections={rank_detections}",
        flush=True,
    )
    return {
        "screenshots": processed,
        "item_detections": item_detections,
        "skill_detections": skill_detections,
        "rank_detections": rank_detections,
    }
