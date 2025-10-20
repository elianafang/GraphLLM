"""Lightweight rule-based extractor for MIRAGE+ graph construction.

The MIRAGE+ research notes describe a rich schema with seven node types and
nine core relationships.  The real system is expected to use an LLM-backed
IE pipeline, but for the open-source reference implementation we provide a
deterministic extractor so that the runtime can materialise a graph even
when the upstream model does not emit structured annotations.  The goal of
this module is therefore *coverage over perfection* – it prefers recalling
useful entities/relations using transparent heuristics instead of brittle
template matching.

The extractor consumes free-form Chinese or English text and derives:

* referenced persons (based on the registered cast list)
* probable locations, objects and temporal expressions
* coarse activity hints such as whether the speaker claims to be at a
  location or to have used an object

The runtime then projects these hints into MIRAGE+ nodes/edges.  Although
the heuristics are intentionally simple, they produce a tangible graph that
developers can inspect, extend and swap out for stronger models later on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Container summarising the entities derived from a text span."""

    persons: Set[str]
    locations: Set[str]
    objects: Set[str]
    times: Set[str]
    used_objects: Set[str]
    claimed_locations: Set[str]
    summary: str


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


class GraphExtractor:
    """Very lightweight regex/keyword based entity extractor."""

    # Location suffix/pattern hints (Chinese + English) that commonly appear
    # in MIRAGE style scripts.
    LOCATION_KEYWORDS = [
        "房间",
        "房",
        "室",
        "厅",
        "馆",
        "园",
        "店",
        "屋",
        "堂",
        "院",
        "巷",
        "楼",
        "甲板",
        "厨房",
        "客厅",
        "花园",
        "办公室",
        "书房",
        "餐厅",
        "旅馆",
        "走廊",
        "浴室",
        "厕所",
        "卧室",
        "库",
        "房屋",
        "庭院",
        "garden",
        "room",
        "hall",
        "office",
        "kitchen",
        "study",
        "garage",
        "balcony",
        "deck",
        "lobby",
        "restaurant",
        "hotel",
        "yard",
        "shed",
        "warehouse",
    ]

    # Object keywords focusing on clues/weapons that usually appear in
    # deduction games.
    OBJECT_KEYWORDS = [
        "刀",
        "剑",
        "匕首",
        "枪",
        "钥匙",
        "锁",
        "信",
        "信件",
        "信封",
        "票",
        "药",
        "毒",
        "瓶",
        "杯",
        "酒",
        "血",
        "指纹",
        "毛",
        "绳",
        "绳子",
        "袋",
        "箱",
        "盒",
        "包",
        "衣",
        "服",
        "鞋",
        "帽",
        "手机",
        "手表",
        "照片",
        "胸牌",
        "御守",
        "牌",
        "卡",
        "日记",
        "手札",
        "信物",
        "护身符",
        "绷带",
        "烟",
        "火",
        "rope",
        "gun",
        "knife",
        "dagger",
        "pistol",
        "rifle",
        "key",
        "note",
        "letter",
        "ticket",
        "bottle",
        "poison",
        "glass",
        "cup",
        "blood",
        "fingerprint",
        "photograph",
        "amulet",
        "charm",
        "card",
        "diary",
        "journal",
        "rope",
        "bag",
        "box",
    ]

    TIME_PATTERNS = [
        re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"),
        re.compile(r"\b(?:[01]?\d|2[0-3])点(?:[0-5]?\d)?分?"),
        re.compile(r"(?:凌晨|清晨|上午|中午|下午|傍晚|晚上|夜里)[\d零一二三四五六七八九十百]*点?"),
        re.compile(r"\b(?:1[0-2]|0?[1-9])(?:[:.][0-5]\d)?\s?(?:[ap]m|[AP]M)\b"),
    ]

    def __init__(self) -> None:
        self.known_persons: Set[str] = set()
        self.known_locations: Set[str] = set()
        self.known_objects: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_persons(self, persons: Iterable[str]) -> None:
        for name in persons:
            cleaned = (name or "").strip()
            if cleaned:
                self.known_persons.add(cleaned)

    def update_locations(self, locations: Iterable[str]) -> None:
        for name in locations:
            cleaned = (name or "").strip()
            if cleaned:
                self.known_locations.add(cleaned)

    def update_objects(self, objects: Iterable[str]) -> None:
        for name in objects:
            cleaned = (name or "").strip()
            if cleaned:
                self.known_objects.add(cleaned)

    def extract(self, text: str, speaker: Optional[str] = None) -> ExtractionResult:
        text = (text or "").strip()
        persons = self._extract_persons(text, speaker)
        locations = self._extract_locations(text)
        objects = self._extract_objects(text)
        times = self._extract_times(text)

        used_objects = {
            obj
            for obj in objects
            if any(token in text for token in (f"使用{obj}", f"用{obj}", f"拿着{obj}", f"拿起{obj}"))
            or any(token in text.lower() for token in (f"use {obj.lower()}", f"using {obj.lower()}", f"with {obj.lower()}"))
        }

        claimed_locations = {
            loc
            for loc in locations
            if any(
                token in text
                for token in (
                    f"在{loc}",
                    f"去{loc}",
                    f"去了{loc}",
                    f"到{loc}",
                    f"回到{loc}",
                    f"前往{loc}",
                )
            )
            or any(
                token.lower() in text.lower()
                for token in (
                    f"at {loc.lower()}",
                    f"in {loc.lower()}",
                    f"to {loc.lower()}",
                )
            )
        }

        summary = self._derive_summary(text)

        return ExtractionResult(
            persons=persons,
            locations=locations,
            objects=objects,
            times=times,
            used_objects=used_objects,
            claimed_locations=claimed_locations,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Bootstrap helpers used by the runtime to ingest scenario files
    # ------------------------------------------------------------------

    def is_probable_location_name(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in text or keyword in lowered for keyword in self.LOCATION_KEYWORDS)

    def guess_locations(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        # Basic Chinese chunk extraction – look for contiguous characters that
        # end with a location keyword.
        for keyword in self.LOCATION_KEYWORDS:
            pattern = re.compile(rf"([\u4e00-\u9fffA-Za-z0-9]+{re.escape(keyword)})")
            for match in pattern.findall(text):
                matches.add(match)

        # Also reuse the explicit location extraction logic.
        matches.update(self._extract_locations(text))
        return matches

    def guess_objects(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        for keyword in self.OBJECT_KEYWORDS:
            pattern = re.compile(rf"([\u4e00-\u9fffA-Za-z0-9]*{re.escape(keyword)})")
            for match in pattern.findall(text):
                cleaned = match.strip()
                if cleaned:
                    matches.add(cleaned)
        matches.update(self._extract_objects(text))
        return matches

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    def _extract_persons(self, text: str, speaker: Optional[str]) -> Set[str]:
        persons = set()
        for name in self.known_persons:
            if not name or name == speaker:
                continue
            if name in text:
                persons.add(name)
        return persons

    def _extract_locations(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        lowered = text.lower()
        for loc in self.known_locations:
            if loc and (loc in text or loc.lower() in lowered):
                matches.add(loc)
        for keyword in self.LOCATION_KEYWORDS:
            pattern = re.compile(rf"([\u4e00-\u9fffA-Za-z0-9]+{re.escape(keyword)})")
            matches.update(pattern.findall(text))
        return {match.strip() for match in matches if match.strip()}

    def _extract_objects(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        lowered = text.lower()
        for obj in self.known_objects:
            if obj and (obj in text or obj.lower() in lowered):
                matches.add(obj)
        for keyword in self.OBJECT_KEYWORDS:
            pattern = re.compile(rf"([\u4e00-\u9fffA-Za-z0-9]*{re.escape(keyword)})")
            matches.update(pattern.findall(text))
        return {match.strip() for match in matches if match.strip()}

    def _extract_times(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        for pattern in self.TIME_PATTERNS:
            matches.update(pattern.findall(text))
        return {match.strip() for match in matches if match.strip()}

    def _derive_summary(self, text: str) -> str:
        if not text:
            return ""
        sentences = re.split(r"[。！？!?]\s*", text)
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                return cleaned[:120]
        return text[:120]

