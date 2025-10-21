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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
    DEFAULT_LOCATION_KEYWORDS = [
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
    DEFAULT_OBJECT_KEYWORDS = [
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

    DEFAULT_LOCATION_STOPWORDS = {
        "这里",
        "那里",
        "哪里",
        "那边",
        "这边",
        "这儿",
        "那儿",
        "我",
        "你",
        "他",
        "她",
        "我们",
        "你们",
        "他们",
        "自己",
        "附近",
    }

    DEFAULT_OBJECT_STOPWORDS = {
        "东西",
        "事情",
        "情况",
        "时候",
        "时间",
        "人",
        "自己",
        "我们",
        "他们",
        "线索",
        "证据",
    }

    DEFAULT_LOCATION_REGEXES: List[Tuple[str, int]] = [
        (r"(?:在|到|去|前往|回到|进入|离开|出现在|停留在|留在|来到|待在)([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})", 0),
        (
            r"\b(?:at|in|inside|into|to|towards|from|entered|left|approached|went to|walked to|arrived at|stayed in|stayed at)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9_\-]*){0,2})",
            re.IGNORECASE,
        ),
    ]

    DEFAULT_OBJECT_REGEXES: List[Tuple[str, int]] = [
        (r"(?:拿着|拿起|拿到|使用|用|带着|握着|发现|找到|捡到|藏着|遗留|留下|丢下|偷走|偷取|携带|摸到|放下)([\u4e00-\u9fffA-Za-z0-9·\-]{1,10})", 0),
        (
            r"\b(?:with|holding|carrying|using|found|picked up|grabbed|took|left|dropped|hid|stole|stashed)\s+(?:a\s+|the\s+)?([A-Za-z0-9][A-Za-z0-9_\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9_\-]*){0,2})",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        self.known_persons: Set[str] = set()
        self.known_locations: Set[str] = set()
        self.known_objects: Set[str] = set()

        config = config or {}

        self.location_keywords: List[str] = []
        self._location_keyword_set: Set[str] = set()
        configured_location_keywords = config.get("location_keywords")
        if configured_location_keywords is None:
            initial_location_keywords = list(self.DEFAULT_LOCATION_KEYWORDS)
        else:
            initial_location_keywords = list(configured_location_keywords)
        for keyword in initial_location_keywords:
            self._add_location_keyword(keyword)
        for keyword in self._iter_clean_entries(config.get("extra_location_keywords")):
            self._add_location_keyword(keyword)

        self.object_keywords: List[str] = []
        self._object_keyword_set: Set[str] = set()
        configured_object_keywords = config.get("object_keywords")
        if configured_object_keywords is None:
            initial_object_keywords = list(self.DEFAULT_OBJECT_KEYWORDS)
        else:
            initial_object_keywords = list(configured_object_keywords)
        for keyword in initial_object_keywords:
            self._add_object_keyword(keyword)
        for keyword in self._iter_clean_entries(config.get("extra_object_keywords")):
            self._add_object_keyword(keyword)

        self.location_stopwords: Set[str] = set(self.DEFAULT_LOCATION_STOPWORDS)
        self.location_stopwords.update(self._iter_clean_entries(config.get("location_stopwords")))
        self.location_stopwords_lower: Set[str] = {word.lower() for word in self.location_stopwords}

        self.object_stopwords: Set[str] = set(self.DEFAULT_OBJECT_STOPWORDS)
        self.object_stopwords.update(self._iter_clean_entries(config.get("object_stopwords")))
        self.object_stopwords_lower: Set[str] = {word.lower() for word in self.object_stopwords}

        location_pattern_entries = config.get("location_patterns")
        if location_pattern_entries is None:
            location_pattern_entries = list(self.DEFAULT_LOCATION_REGEXES)
        self.location_patterns: List[re.Pattern] = []
        for entry in location_pattern_entries:
            parsed = self._coerce_pattern_entry(entry)
            if not parsed:
                continue
            pattern, flags = parsed
            self.location_patterns.append(re.compile(pattern, flags))
        for entry in config.get("extra_location_patterns", []):
            parsed = self._coerce_pattern_entry(entry)
            if not parsed:
                continue
            pattern, flags = parsed
            self.location_patterns.append(re.compile(pattern, flags))

        object_pattern_entries = config.get("object_patterns")
        if object_pattern_entries is None:
            object_pattern_entries = list(self.DEFAULT_OBJECT_REGEXES)
        self.object_patterns: List[re.Pattern] = []
        for entry in object_pattern_entries:
            parsed = self._coerce_pattern_entry(entry)
            if not parsed:
                continue
            pattern, flags = parsed
            self.object_patterns.append(re.compile(pattern, flags))
        for entry in config.get("extra_object_patterns", []):
            parsed = self._coerce_pattern_entry(entry)
            if not parsed:
                continue
            pattern, flags = parsed
            self.object_patterns.append(re.compile(pattern, flags))

    # ------------------------------------------------------------------
    # Internal configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_clean_entries(values: Optional[Iterable[Any]]) -> Iterable[str]:
        if not values:
            return []
        cleaned: List[str] = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _parse_pattern_config(entry: Any) -> Optional[Tuple[str, int]]:
        if isinstance(entry, str):
            return entry, 0
        if isinstance(entry, dict) and "pattern" in entry:
            flags = 0
            if entry.get("ignore_case", True):
                flags |= re.IGNORECASE
            if entry.get("multiline"):
                flags |= re.MULTILINE
            if entry.get("dotall"):
                flags |= re.DOTALL
            return str(entry["pattern"]), flags
        if isinstance(entry, tuple) and len(entry) == 2:
            pattern, flags = entry
            if isinstance(pattern, str):
                return pattern, int(flags)
        return None

    @classmethod
    def _coerce_pattern_entry(cls, entry: Any) -> Optional[Tuple[str, int]]:
        return cls._parse_pattern_config(entry)

    def _add_location_keyword(self, keyword: str) -> None:
        cleaned = (keyword or "").strip()
        if not cleaned:
            return
        if cleaned not in self._location_keyword_set:
            self._location_keyword_set.add(cleaned)
            self.location_keywords.append(cleaned)

    def _add_object_keyword(self, keyword: str) -> None:
        cleaned = (keyword or "").strip()
        if not cleaned:
            return
        if cleaned not in self._object_keyword_set:
            self._object_keyword_set.add(cleaned)
            self.object_keywords.append(cleaned)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
                self._learn_location_keywords({cleaned})

    def update_objects(self, objects: Iterable[str]) -> None:
        for name in objects:
            cleaned = (name or "").strip()
            if cleaned:
                self.known_objects.add(cleaned)
                self._learn_object_keywords({cleaned})

    def extract(self, text: str, speaker: Optional[str] = None) -> ExtractionResult:
        text = (text or "").strip()
        persons = self._extract_persons(text, speaker)
        locations = self._extract_locations(text)
        objects = self._extract_objects(text)
        times = self._extract_times(text)

        self._learn_location_keywords(locations)
        self._learn_object_keywords(objects)

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
        text = (text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        for known in self.known_locations:
            if not known:
                continue
            if known == text or known.lower() == lowered or known in text:
                return True
        return any(keyword and (keyword in text or keyword in lowered) for keyword in self.location_keywords)

    def guess_locations(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        for keyword in self.location_keywords:
            matches.update(self._match_keyword_sequences(text, keyword))
        for pattern in self.location_patterns:
            matches.update(self._collect_pattern_matches(pattern, text))
        matches.update(self._extract_locations(text))
        cleaned: Set[str] = set()
        for candidate in matches:
            cleaned_loc = self._clean_location_candidate(candidate)
            if cleaned_loc:
                cleaned.add(cleaned_loc)
        return cleaned

    def guess_objects(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        for keyword in self.object_keywords:
            matches.update(self._match_keyword_sequences(text, keyword))
        for pattern in self.object_patterns:
            matches.update(self._collect_pattern_matches(pattern, text))
        matches.update(self._extract_objects(text))
        cleaned: Set[str] = set()
        for candidate in matches:
            cleaned_obj = self._clean_object_candidate(candidate)
            if cleaned_obj:
                cleaned.add(cleaned_obj)
        return cleaned

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

    def _match_keyword_sequences(self, text: str, keyword: str) -> Set[str]:
        keyword = (keyword or "").strip()
        if not keyword:
            return set()
        ignore_case = bool(re.search(r"[A-Za-z]", keyword))
        pattern = re.compile(
            rf"((?:[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9\s\-'·]{{0,12}})?{re.escape(keyword)})",
            re.IGNORECASE if ignore_case else 0,
        )
        return {match.strip() for match in pattern.findall(text)}

    def _collect_pattern_matches(self, pattern: re.Pattern, text: str) -> Set[str]:
        results: Set[str] = set()
        for match in pattern.findall(text):
            if isinstance(match, tuple):
                if match:
                    results.add(match[0])
            else:
                results.add(match)
        return results

    def _clean_candidate(
        self,
        candidate: str,
        stopwords: Set[str],
        stopwords_lower: Set[str],
        *,
        min_length: int = 1,
    ) -> Optional[str]:
        if not candidate:
            return None
        cleaned = str(candidate).strip()
        if not cleaned:
            return None
        cleaned = cleaned.strip(".,!?;:、，。！？；：\"'“”‘’()（）[]【】")
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned or len(cleaned) < min_length:
            return None
        lowered = cleaned.lower()
        if cleaned in stopwords or lowered in stopwords_lower:
            return None
        return cleaned

    def _clean_location_candidate(self, candidate: str) -> Optional[str]:
        cleaned = self._clean_candidate(
            candidate,
            self.location_stopwords,
            self.location_stopwords_lower,
            min_length=2,
        )
        if not cleaned:
            return None
        cleaned = re.sub(r"(?:里|裡|里面|內|內部|当中|之中|附近|左右)$", "", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if cleaned in self.location_stopwords or lowered in self.location_stopwords_lower:
            return None
        return cleaned

    def _clean_object_candidate(self, candidate: str) -> Optional[str]:
        cleaned = self._clean_candidate(
            candidate,
            self.object_stopwords,
            self.object_stopwords_lower,
            min_length=1,
        )
        if not cleaned:
            return None
        cleaned = re.sub(r"(?:的|了)$", "", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if cleaned in self.object_stopwords or lowered in self.object_stopwords_lower:
            return None
        return cleaned

    def _learn_location_keywords(self, locations: Iterable[str]) -> None:
        for loc in locations:
            if not loc:
                continue
            cleaned = loc.strip()
            if not cleaned:
                continue
            if re.search(r"[A-Za-z]", cleaned):
                last_token = cleaned.split()[-1].lower()
                if last_token and last_token not in self.location_stopwords_lower:
                    self._add_location_keyword(last_token)
            elif len(cleaned) >= 2:
                suffix = cleaned[-1]
                if suffix and suffix not in self.location_stopwords and suffix not in self._location_keyword_set:
                    self._add_location_keyword(suffix)

    def _learn_object_keywords(self, objects: Iterable[str]) -> None:
        for obj in objects:
            if not obj:
                continue
            cleaned = obj.strip()
            if not cleaned:
                continue
            if re.search(r"[A-Za-z]", cleaned):
                last_token = cleaned.split()[-1].lower()
                if last_token and last_token not in self.object_stopwords_lower:
                    self._add_object_keyword(last_token)
            elif len(cleaned) >= 1:
                suffix = cleaned[-1]
                if suffix and suffix not in self.object_stopwords and suffix not in self._object_keyword_set:
                    self._add_object_keyword(suffix)

    def _extract_locations(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        lowered = text.lower()
        for loc in self.known_locations:
            if loc and (loc in text or loc.lower() in lowered):
                matches.add(loc)
        for keyword in self.location_keywords:
            matches.update(self._match_keyword_sequences(text, keyword))
        for pattern in self.location_patterns:
            matches.update(self._collect_pattern_matches(pattern, text))
        cleaned: Set[str] = set()
        for match in matches:
            cleaned_loc = self._clean_location_candidate(match)
            if cleaned_loc:
                cleaned.add(cleaned_loc)
        return cleaned

    def _extract_objects(self, text: str) -> Set[str]:
        matches: Set[str] = set()
        lowered = text.lower()
        for obj in self.known_objects:
            if obj and (obj in text or obj.lower() in lowered):
                matches.add(obj)
        for keyword in self.object_keywords:
            matches.update(self._match_keyword_sequences(text, keyword))
        for pattern in self.object_patterns:
            matches.update(self._collect_pattern_matches(pattern, text))
        cleaned: Set[str] = set()
        for match in matches:
            cleaned_obj = self._clean_object_candidate(match)
            if cleaned_obj:
                cleaned.add(cleaned_obj)
        return cleaned

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