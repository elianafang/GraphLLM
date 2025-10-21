import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math

from graph_extractor import ExtractionResult, GraphExtractor


@dataclass
class FactRecord:
    source_id: str
    speaker: str
    round_index: Optional[int]
    text: str
    source_kind: str
    persons: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    times: Set[str] = field(default_factory=set)
    objects: Set[str] = field(default_factory=set)
    confidence: float = 0.5
    event_id: Optional[str] = None
    veracity: str = "claim"


@dataclass
class GraphNode:
    id: str
    type: str
    name: str
    confidence: float
    source_agent: Optional[str]
    source_round: Optional[int]
    provenance: str
    text_span: Optional[str]
    last_updated: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    type: str
    source: str
    target: str
    confidence: float
    source_agent: Optional[str]
    source_round: Optional[int]
    provenance: str
    text_span: Optional[str]
    last_updated: str
    attributes: Dict[str, Any] = field(default_factory=dict)


class MiragePlusRuntime:
    """Runtime manager that tracks MIRAGE+ graph state, conflicts and plans."""

    def __init__(self, script_name: str, log_dir: Path, config: Any):
        self.script_name = script_name
        self.log_dir = Path(log_dir)
        self.config = config

        self.mplus_config: Dict[str, Any] = getattr(config, "MPLUS", {}) or {}
        self.thresholds = self.mplus_config.get("thresholds", {})
        self.planner_config = self.mplus_config.get("planner", {})
        self.summary_config = self.mplus_config.get("summary", {})
        extractor_config = self.mplus_config.get("extractor", {})
        graph_config = self.mplus_config.get("graph", {}) or {}

        default_nodes = [
            "Person",
            "Event",
            "Object",
            "Location",
            "Time",
            "Clue",
            "Statement",
        ]
        default_edges = [
            "was_at",
            "used",
            "involves",
            "occurs_in",
            "happens_at",
            "refers_to",
            "supports",
            "contradicts",
            "about",
        ]
        self.allowed_node_types: Set[str] = set(graph_config.get("node_types", default_nodes))
        self.allowed_edge_types: Set[str] = set(graph_config.get("edge_types", default_edges))

        self.graph_path = self.log_dir / "graph_state.json"
        self.contradictions_path = self.log_dir / "contradictions.json"
        self.plan_path = self.log_dir / "plan_suggestions.json"
        self.votes_path = self.log_dir / "votes.json"
        self.evaluation_path = self.log_dir / "evaluation.json"
        self.debug_trace_path = self.log_dir / "debug_trace.txt"

        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.graph_state: Dict[str, Any] = {}
        self.contradictions: List[Dict[str, Any]] = []
        self.plan_log: Dict[int, Dict[str, Any]] = {}
        self.current_stage: Optional[str] = None
        self.current_round: Optional[int] = None
        self.candidate_cache: Dict[str, Dict[str, Any]] = {}
        self.latest_plan: Optional[Dict[str, Any]] = None
        self.statement_history: List[Dict[str, Any]] = []
        self.round_message_buffer: List[str] = []
        self.victory_mrr: float = 0.0
        self.fact_records: List["FactRecord"] = []
        self._conflict_counter: int = 0
        self._conflict_index: Set[frozenset[str]] = set()

        self._edge_index: Dict[Tuple[str, str, str], str] = {}
        self._id_counters: Dict[str, int] = {
            "Person": 0,
            "Statement": 0,
            "Clue": 0,
            "Event": 0,
            "Object": 0,
            "Location": 0,
            "Time": 0
        }

        self.extractor = GraphExtractor(config=extractor_config)
        self._statement_event: Dict[str, str] = {}
        self.known_locations: Set[str] = set()
        self.known_objects: Set[str] = set()
        self.known_times: Set[str] = set()

        # Bootstrap log files so downstream consumers always find them.
        self._ensure_file(self.graph_path, {})
        self._ensure_file(self.contradictions_path, self.contradictions)
        self._ensure_file(self.plan_path, [])
        self._ensure_file(self.votes_path, {})
        self._ensure_file(self.evaluation_path, {})
        self._ensure_debug_file()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def update_stage(self, stage: Optional[str]) -> None:
        if stage == self.current_stage:
            return
        if self.current_stage == "conversation":
            self._finalize_round()
        self.current_stage = stage
        if stage != "conversation":
            self.current_round = None

    def register_persons(self, persons: List[str]) -> None:
        for name in persons:
            self._ensure_person_node(name)
        self.extractor.update_persons(persons)
        # Emit initial snapshot for tooling to consume the cast list.
        self._write_graph_snapshot(round_index=0)

    def register_locations(self, locations: Iterable[str]) -> None:
        for location in locations:
            location = location.strip()
            if not location or location in self.known_locations:
                continue
            self.known_locations.add(location)
            self.extractor.update_locations([location])
            self._ensure_location_node(
                location,
                confidence=0.7,
                provenance="bootstrap",
                timestamp=datetime.utcnow().isoformat() + "Z",
                source_agent=None,
                source_round=None,
                veracity="groundtruth",
            )

    def register_objects(self, objects: Iterable[str]) -> None:
        for obj in objects:
            obj = obj.strip()
            if not obj or obj in self.known_objects:
                continue
            self.known_objects.add(obj)
            self.extractor.update_objects([obj])
            self._ensure_object_node(
                obj,
                confidence=0.65,
                provenance="bootstrap",
                timestamp=datetime.utcnow().isoformat() + "Z",
                source_agent=None,
                source_round=None,
                veracity="groundtruth",
            )

    def bootstrap_from_clues(self, clues: Dict[str, Any]) -> None:
        locations: Set[str] = set()
        objects: Set[str] = set()
        times: Set[str] = set()
        for stage_data in clues.values():
            if not isinstance(stage_data, dict):
                continue
            for key, entries in stage_data.items():
                key_str = str(key).strip()
                if key_str:
                    if self.extractor.is_probable_location_name(key_str):
                        locations.add(key_str)
                if isinstance(entries, list):
                    for text in entries:
                        extraction = self.extractor.extract(str(text))
                        locations.update(extraction.locations)
                        objects.update(extraction.objects)
                        times.update(extraction.times)
                        guessed_locations = self.extractor.guess_locations(str(text))
                        locations.update(guessed_locations)
                        guessed_objects = self.extractor.guess_objects(str(text))
                        objects.update(guessed_objects)
        self.register_locations(sorted(locations))
        self.register_objects(sorted(objects))
        self.known_times.update(times)
        for time_value in sorted(times):
            self._ensure_time_node(
                time_value,
                confidence=0.6,
                provenance="bootstrap",
                timestamp=datetime.utcnow().isoformat() + "Z",
                source_agent=None,
                source_round=None,
                veracity="groundtruth",
            )

    def bootstrap_from_scripts(self, scripts: Dict[str, Any]) -> None:
        locations: Set[str] = set()
        objects: Set[str] = set()
        times: Set[str] = set()
        for content in scripts.values():
            if not isinstance(content, dict):
                continue
            for value in content.values():
                if not isinstance(value, str):
                    continue
                extraction = self.extractor.extract(value)
                locations.update(extraction.locations)
                objects.update(extraction.objects)
                times.update(extraction.times)
        self.register_locations(sorted(locations))
        self.register_objects(sorted(objects))
        self.known_times.update(times)
        for time_value in sorted(times):
            self._ensure_time_node(
                time_value,
                confidence=0.55,
                provenance="bootstrap",
                timestamp=datetime.utcnow().isoformat() + "Z",
                source_agent=None,
                source_round=None,
                veracity="groundtruth",
            )

    def record_message(
        self,
        *,
        user: str,
        text: str,
        template: str,
        model: Optional[str],
        stage: Optional[str],
        round_index: Optional[int],
        mp_extras: Optional[Dict[str, Any]] = None
    ) -> None:
        if stage != self.current_stage:
            self.update_stage(stage)

        if stage == "conversation":
            if round_index is None:
                round_index = 0
            if self.current_round is None or round_index > self.current_round:
                self._finalize_round()
                self.start_round(round_index)
        elif stage == "self_introduction":
            round_index = 0
            if self.current_round is None:
                self.start_round(round_index)
        else:
            round_index = round_index if round_index is not None else self.current_round

        timestamp = datetime.utcnow().isoformat() + "Z"
        provenance = self._infer_provenance(template)

        if template == "clue":
            self._record_clue_node(user, text, round_index, timestamp, provenance)
        elif user != "Env":
            self._record_statement_node(
                user=user,
                text=text,
                round_index=round_index,
                model=model,
                timestamp=timestamp,
                provenance=provenance,
                mp_extras=mp_extras,
            )
        else:
            self._append_debug(f"[{timestamp}] ENV<{template}>: {text[:200]}")

        if stage == "conversation" and text:
            self.round_message_buffer.append(text)

    def update_candidates(self, candidates: Dict[str, Dict[str, Any]]) -> None:
        self.candidate_cache = json.loads(json.dumps(candidates))
        if self.current_stage == "conversation" and self.current_round is not None:
            self._update_plan_for_round(self.current_round)

    def render_context(
        self,
        role: Optional[str] = None,
        template: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        package = self.build_context_package(role=role, template=template, model=model)
        if not package:
            return ""
        return "\n[MIRAGE+ CONTEXT]\n" + json.dumps(package, ensure_ascii=False, indent=2)

    def build_context_package(
        self,
        *,
        role: Optional[str],
        template: Optional[str],
        model: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if self.current_stage is None:
            return None

        dialogue_summary = self._summarize_dialogue()
        facts = self._collect_recent_facts()
        conflicts = [
            f"{c['id']}: {c.get('type', 'Unknown')} (severity={c.get('severity', 0):.2f})"
            for c in self.contradictions
        ]

        plan_hint = self.latest_plan or {
            "targets": [],
            "action": None,
            "query": None,
            "expected_info_gain": 0.0,
            "rationale": "Insufficient data to plan."
        }
        if plan_hint:
            plan_hint = {
                **plan_hint,
                "max_questions_per_round": self.planner_config.get("max_questions_per_round"),
            }

        package = {
            "round": self.current_round,
            "role": role,
            "dialogue_summary": dialogue_summary,
            "graph_summary": {
                "facts": facts,
                "event_chain": self._collect_event_chain(),
            },
            "conflict_summary": conflicts,
            "plan_hint": plan_hint,
            "prompt_template": template,
            "api_model": model,
            "planner_config": {
                "topk": self.planner_config.get("topk"),
                "max_questions_per_round": self.planner_config.get("max_questions_per_round"),
            },
        }
        return package

    def record_votes(self, ranking: List[str], voted: Optional[str], culprits: List[str]) -> None:
        payload = {
            "doubt_ranking": ranking,
            "voted_culprit": voted
        }
        self._write_json(self.votes_path, payload)
        self._append_debug(f"Votes recorded: {payload}")
        self._update_victory_metric(ranking, culprits)

    def record_evaluation(self, result: Dict[str, Any]) -> None:
        metrics = {
            "fii": result.get("fii"),
            "clue": result.get("clue"),
            "llm": result.get("llm"),
            "rouge": result.get("rouge"),
            "victory_mrr": self.victory_mrr,
        }
        valid_scores = [metrics[key] for key in ["llm", "rouge", "fii", "victory_mrr"] if metrics[key] is not None]
        metrics["overall"] = sum(valid_scores) / len(valid_scores) if valid_scores else None
        payload = {
            "script": self.script_name,
            "metrics": metrics
        }
        self._write_json(self.evaluation_path, payload)
        self._append_debug(f"Evaluation recorded: {payload}")

    def close(self) -> None:
        self._finalize_round()
        if self.plan_log:
            ordered_plans = [self.plan_log[k] for k in sorted(self.plan_log.keys())]
            self._write_json(self.plan_path, ordered_plans)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def start_round(self, round_index: int) -> None:
        self.current_round = round_index
        self.round_message_buffer = []
        self._append_debug(f"Start round {round_index}")
        self._update_plan_for_round(round_index)

    def _finalize_round(self) -> None:
        if self.current_round is None:
            return
        self._append_debug(f"Finalize round {self.current_round}")
        self._write_graph_snapshot(self.current_round)
        self.current_round = None
        self.round_message_buffer = []

    def _write_graph_snapshot(self, round_index: int) -> None:
        snapshot = {
            "round": round_index,
            "nodes": [self._node_to_dict(node) for node in self.nodes.values()],
            "edges": [self._edge_to_dict(edge) for edge in self.edges.values()],
            "snapshot_id": f"G_r{round_index}"
        }
        self.graph_state = snapshot
        self._write_json(self.graph_path, snapshot)

    def _flush_graph(self, round_index: Optional[int]) -> None:
        """Persist the current in-memory graph after incremental updates."""
        if round_index is None:
            round_index = self.current_round
        if round_index is None:
            round_index = 0
        self._write_graph_snapshot(round_index)

    def _record_statement_node(
        self,
        *,
        user: str,
        text: str,
        round_index: Optional[int],
        model: Optional[str],
        timestamp: str,
        provenance: str,
        mp_extras: Optional[Dict[str, Any]]
    ) -> None:
        self._ensure_person_node(user)
        statement_id = self._next_id("Statement")
        thought, response = self._split_response(text)
        node = GraphNode(
            id=statement_id,
            type="Statement",
            name=f"{user}_statement_{statement_id}",
            confidence=mp_extras.get("confidence", 0.5) if mp_extras else 0.5,
            source_agent=user,
            source_round=round_index,
            provenance=provenance,
            text_span=response or text,
            last_updated=timestamp,
            attributes={
                "model": model,
                "thought": thought,
                "mp_extras": mp_extras or {},
                "veracity": "claim",
            }
        )
        self.nodes[node.id] = node
        self.statement_history.append({
            "round": round_index,
            "speaker": user,
            "response": response or text,
            "statement_id": statement_id,
        })
        edge = GraphEdge(
            id=self._next_edge_id("refers_to"),
            type="refers_to",
            source=node.id,
            target=self._person_node_id(user),
            confidence=node.confidence,
            source_agent=user,
            source_round=round_index,
            provenance=provenance,
            text_span=response or text,
            last_updated=timestamp,
            attributes=self._edge_style("statement"),
        )
        self._add_edge(edge)

        extraction, event_id = self._process_extraction(
            speaker=user,
            text=response or text,
            round_index=round_index,
            provenance=provenance,
            timestamp=timestamp,
            source_node=node,
            source_kind="statement",
            mp_extras=mp_extras,
        )
        if extraction:
            self._register_fact(
                source_node=node,
                extraction=extraction,
                event_id=event_id,
                source_kind="statement",
                mp_extras=mp_extras,
            )
            self._evaluate_conflicts(
                statement=node,
                extraction=extraction,
                timestamp=timestamp,
            )

        self._flush_graph(round_index)

    def _record_clue_node(
        self,
        user: str,
        text: str,
        round_index: Optional[int],
        timestamp: str,
        provenance: str
    ) -> None:
        clue_id = self._next_id("Clue")
        node = GraphNode(
            id=clue_id,
            type="Clue",
            name=user,
            confidence=1.0,
            source_agent=user,
            source_round=round_index,
            provenance=provenance,
            text_span=text,
            last_updated=timestamp,
            attributes={"veracity": "groundtruth"},
        )
        self.nodes[node.id] = node
        extraction, event_id = self._process_extraction(
            speaker=user,
            text=text,
            round_index=round_index,
            provenance=provenance,
            timestamp=timestamp,
            source_node=node,
            source_kind="clue",
            mp_extras=None,
        )
        if extraction:
            self._register_fact(
                source_node=node,
                extraction=extraction,
                event_id=event_id,
                source_kind="clue",
                mp_extras=None,
            )

        self._flush_graph(round_index)

    def _process_extraction(
        self,
        *,
        speaker: str,
        text: str,
        round_index: Optional[int],
        provenance: str,
        timestamp: str,
        source_node: Optional[GraphNode],
        source_kind: str,
        mp_extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not text:
            return None, None
        extraction = self.extractor.extract(text, speaker)
        if not any([extraction.persons, extraction.locations, extraction.objects, extraction.times]):
            return None, None

        event_id = None
        style = self._edge_style(source_kind)
        node_veracity = "groundtruth" if source_kind == "clue" else "claim"
        source_id = source_node.id if source_node else None
        if source_kind == "statement" and source_node is not None:
            event_id = self._ensure_event_for_statement(
                source_node,
                extraction,
                round_index,
                provenance,
                timestamp,
            )
            self._link_statement_to_event(source_node.id, event_id, extraction, provenance, timestamp)

        for person in extraction.persons:
            target_id = self._person_node_id(person)
            if source_id and source_kind == "statement":
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("refers_to"),
                        type="refers_to",
                        source=source_id,
                        target=target_id,
                        confidence=0.6,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )
            if event_id:
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("involves"),
                        type="involves",
                        source=event_id,
                        target=target_id,
                        confidence=0.55,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )

        for location in extraction.locations:
            location_node = self._ensure_location_node(
                location,
                confidence=0.6,
                provenance=provenance,
                timestamp=timestamp,
                source_agent=speaker,
                source_round=round_index,
                veracity=node_veracity,
            )
            if event_id:
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("occurs_in"),
                        type="occurs_in",
                        source=event_id,
                        target=location_node.id,
                        confidence=0.6,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )
            if source_kind == "statement" and location in extraction.claimed_locations and speaker in self.extractor.known_persons:
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("was_at"),
                        type="was_at",
                        source=self._person_node_id(speaker),
                        target=location_node.id,
                        confidence=0.6,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )

        for obj in extraction.objects:
            object_node = self._ensure_object_node(
                obj,
                confidence=0.55,
                provenance=provenance,
                timestamp=timestamp,
                source_agent=speaker,
                source_round=round_index,
                veracity=node_veracity,
            )
            if source_id and source_kind == "statement":
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("refers_to"),
                        type="refers_to",
                        source=source_id,
                        target=object_node.id,
                        confidence=0.5,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes={**style, "relation_origin": "refers_to"},
                    )
                )
            if source_id and source_kind == "clue":
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("about"),
                        type="about",
                        source=source_id,
                        target=object_node.id,
                        confidence=0.65,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes={**style, "relation_origin": "about"},
                    )
                )
            if event_id:
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("involves"),
                        type="involves",
                        source=event_id,
                        target=object_node.id,
                        confidence=0.5,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )
            if source_kind == "statement" and obj in extraction.used_objects and speaker in self.extractor.known_persons:
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("used"),
                        type="used",
                        source=self._person_node_id(speaker),
                        target=object_node.id,
                        confidence=0.6,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )

        for time_str in extraction.times:
            time_node = self._ensure_time_node(
                time_str,
                confidence=0.55,
                provenance=provenance,
                timestamp=timestamp,
                source_agent=speaker,
                source_round=round_index,
                veracity=node_veracity,
            )
            if event_id:
                self._add_edge(
                    GraphEdge(
                        id=self._next_edge_id("happens_at"),
                        type="happens_at",
                        source=event_id,
                        target=time_node.id,
                        confidence=0.55,
                        source_agent=speaker,
                        source_round=round_index,
                        provenance=provenance,
                        text_span=text,
                        last_updated=timestamp,
                        attributes=dict(style),
                    )
                )

        return extraction, event_id

    def _ensure_event_for_statement(
        self,
        statement_node: GraphNode,
        extraction: ExtractionResult,
        round_index: Optional[int],
        provenance: str,
        timestamp: str,
    ) -> str:
        if statement_node.id in self._statement_event:
            return self._statement_event[statement_node.id]
        event_id = self._next_id("Event")
        participants = set(extraction.persons)
        if statement_node.source_agent:
            participants.add(statement_node.source_agent)
        times = sorted(extraction.times)
        locations = sorted(extraction.locations)
        objects = sorted(extraction.objects)
        summary = extraction.summary or statement_node.text_span or statement_node.name
        label_parts: List[str] = []
        if participants:
            label_parts.append("/".join(sorted(participants)))
        if times:
            label_parts.append("@" + "/".join(times))
        if locations:
            label_parts.append("在" + "/".join(locations))
        if summary:
            label_parts.append(self._truncate_text(summary, 80))
        event_name = " ".join(part for part in label_parts if part) or f"Event {event_id}"
        event_node = GraphNode(
            id=event_id,
            type="Event",
            name=event_name,
            confidence=0.55,
            source_agent=statement_node.source_agent,
            source_round=round_index,
            provenance=provenance,
            text_span=summary,
            last_updated=timestamp,
            attributes={
                "source_statement": statement_node.id,
                "summary": summary,
                "participants": sorted(participants),
                "times": times,
                "locations": locations,
                "objects": objects,
                "veracity": "claim",
            },
        )
        self.nodes[event_node.id] = event_node
        self._statement_event[statement_node.id] = event_id
        return event_id

    def _link_statement_to_event(
        self,
        statement_id: str,
        event_id: str,
        extraction: ExtractionResult,
        provenance: str,
        timestamp: str,
    ) -> None:
        statement_node = self.nodes.get(statement_id)
        if statement_node:
            refers_edge = GraphEdge(
                id=self._next_edge_id("refers_to"),
                type="refers_to",
                source=statement_id,
                target=event_id,
                confidence=statement_node.confidence,
                source_agent=statement_node.source_agent,
                source_round=statement_node.source_round,
                provenance=provenance,
                text_span=extraction.summary,
                last_updated=timestamp,
                attributes=self._edge_style("statement"),
            )
            self._add_edge(refers_edge)
        edge = GraphEdge(
            id=self._next_edge_id("supports"),
            type="supports",
            source=statement_id,
            target=event_id,
            confidence=0.55,
            source_agent=self.nodes[statement_id].source_agent,
            source_round=self.nodes[statement_id].source_round,
            provenance=provenance,
            text_span=extraction.summary,
            last_updated=timestamp,
            attributes={**self._edge_style("statement"), "relation": "statement_reports_event"},
        )
        self._add_edge(edge)

    def _register_fact(
        self,
        *,
        source_node: GraphNode,
        extraction: ExtractionResult,
        event_id: Optional[str],
        source_kind: str,
        mp_extras: Optional[Dict[str, Any]],
    ) -> None:
        locations = set(extraction.claimed_locations or extraction.locations)
        times = set(extraction.times)
        objects = set(extraction.objects)
        persons = set(extraction.persons)
        if source_node.source_agent:
            persons.add(source_node.source_agent)
        confidence = source_node.confidence
        if mp_extras:
            extra_conf = mp_extras.get("confidence")
            if isinstance(extra_conf, (int, float)):
                confidence = max(0.0, min(1.0, float(extra_conf)))
        veracity = "groundtruth" if source_kind == "clue" else "claim"
        fact = FactRecord(
            source_id=source_node.id,
            speaker=source_node.source_agent or source_node.name,
            round_index=source_node.source_round,
            text=source_node.text_span or extraction.summary,
            source_kind=source_kind,
            persons=persons,
            locations=locations,
            times=times,
            objects=objects,
            confidence=confidence,
            event_id=event_id,
            veracity=veracity,
        )
        self.fact_records.append(fact)
        if source_kind == "statement":
            for item in reversed(self.statement_history):
                if item.get("statement_id") == source_node.id:
                    item["locations"] = sorted(locations)
                    item["times"] = sorted(times)
                    item["objects"] = sorted(objects)
                    item["persons"] = sorted(persons)
                    item["veracity"] = veracity
                    break

    def _evaluate_conflicts(
        self,
        *,
        statement: GraphNode,
        extraction: ExtractionResult,
        timestamp: str,
    ) -> None:
        fact = next((f for f in self.fact_records if f.source_id == statement.id and f.source_kind == "statement"), None)
        if fact is None:
            return
        if not fact.locations or not fact.times:
            return
        for prior in self.fact_records:
            if prior.source_kind != "statement":
                continue
            if prior.source_id == fact.source_id:
                continue
            if prior.speaker != fact.speaker:
                continue
            if not prior.locations or not prior.times:
                continue
            overlapping_times = fact.times & prior.times
            if not overlapping_times:
                continue
            if fact.locations & prior.locations:
                continue
            key = frozenset({fact.source_id, prior.source_id})
            if key in self._conflict_index:
                continue
            severity = max(fact.confidence, prior.confidence)
            severity = max(severity, float(self.thresholds.get("conflict_severity", 0.6)))
            severity = min(severity + 0.1, 1.0)
            description = (
                f"{fact.speaker} 在 {', '.join(sorted(overlapping_times))} "
                f"分别声称位于 {', '.join(sorted(prior.locations))} 与 {', '.join(sorted(fact.locations))}"
            )
            conflict = self._raise_conflict(
                kind="TimeLocation",
                statement_ids=[prior.source_id, fact.source_id],
                speakers=[fact.speaker],
                times=sorted(overlapping_times),
                locations={
                    prior.source_id: sorted(prior.locations),
                    fact.source_id: sorted(fact.locations),
                },
                severity=severity,
                description=description,
                timestamp=timestamp,
            )
            if conflict:
                self._conflict_index.add(key)
                self._add_contradiction_edges(
                    prior.source_id,
                    fact.source_id,
                    severity,
                    timestamp,
                )

    def _raise_conflict(
        self,
        *,
        kind: str,
        statement_ids: List[str],
        speakers: List[str],
        times: List[str],
        locations: Dict[str, List[str]],
        severity: float,
        description: str,
        timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        if not statement_ids:
            return None
        self._conflict_counter += 1
        conflict_id = f"C{self._conflict_counter}"
        payload = {
            "id": conflict_id,
            "round_created": self.current_round,
            "type": kind,
            "members": {
                "statements": statement_ids,
                "speakers": list(sorted(set(speakers))),
                "times": times,
                "locations": locations,
            },
            "severity": round(float(severity), 3),
            "status": "open",
            "description": description,
            "last_updated": timestamp,
        }
        self.contradictions.append(payload)
        self._write_json(self.contradictions_path, self.contradictions)
        self._append_debug(f"Conflict raised {conflict_id}: {description}")
        return payload

    def _add_contradiction_edges(
        self,
        statement_a: str,
        statement_b: str,
        severity: float,
        timestamp: str,
    ) -> None:
        node_a = self.nodes.get(statement_a)
        node_b = self.nodes.get(statement_b)
        if not node_a or not node_b:
            return
        for source, target in ((statement_a, statement_b), (statement_b, statement_a)):
            owner = self.nodes.get(source)
            edge = GraphEdge(
                id=self._next_edge_id("contradicts"),
                type="contradicts",
                source=source,
                target=target,
                confidence=round(float(severity), 3),
                source_agent=owner.source_agent if owner else None,
                source_round=owner.source_round if owner else None,
                provenance="conflict_detection",
                text_span=None,
                last_updated=timestamp,
                attributes={**self._edge_style("inference"), "reason": "time_location_conflict"},
            )
            self._add_edge(edge)

    def _ensure_location_node(
        self,
        name: str,
        *,
        confidence: float,
        provenance: str,
        timestamp: str,
        source_agent: Optional[str],
        source_round: Optional[int],
        veracity: str,
    ) -> GraphNode:
        node_id = self._location_node_id(name)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_updated = timestamp
            node.confidence = max(node.confidence, confidence)
            if node.attributes is None:
                node.attributes = {}
            existing = node.attributes.get("veracity")
            if existing != "groundtruth" and veracity:
                node.attributes["veracity"] = veracity
            return node
        node = GraphNode(
            id=node_id,
            type="Location",
            name=name,
            confidence=confidence,
            source_agent=source_agent,
            source_round=source_round,
            provenance=provenance,
            text_span=None,
            last_updated=timestamp,
            attributes={"veracity": veracity} if veracity else {},
        )
        self.nodes[node.id] = node
        self.known_locations.add(name)
        self.extractor.update_locations([name])
        return node

    def _ensure_object_node(
        self,
        name: str,
        *,
        confidence: float,
        provenance: str,
        timestamp: str,
        source_agent: Optional[str],
        source_round: Optional[int],
        veracity: str,
    ) -> GraphNode:
        node_id = self._object_node_id(name)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_updated = timestamp
            node.confidence = max(node.confidence, confidence)
            if node.attributes is None:
                node.attributes = {}
            existing = node.attributes.get("veracity")
            if existing != "groundtruth" and veracity:
                node.attributes["veracity"] = veracity
            return node
        node = GraphNode(
            id=node_id,
            type="Object",
            name=name,
            confidence=confidence,
            source_agent=source_agent,
            source_round=source_round,
            provenance=provenance,
            text_span=None,
            last_updated=timestamp,
            attributes={"veracity": veracity} if veracity else {},
        )
        self.nodes[node.id] = node
        self.known_objects.add(name)
        self.extractor.update_objects([name])
        return node

    def _ensure_time_node(
        self,
        value: str,
        *,
        confidence: float,
        provenance: str,
        timestamp: str,
        source_agent: Optional[str],
        source_round: Optional[int],
        veracity: str,
    ) -> GraphNode:
        node_id = self._time_node_id(value)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_updated = timestamp
            node.confidence = max(node.confidence, confidence)
            if node.attributes is None:
                node.attributes = {}
            existing = node.attributes.get("veracity")
            if existing != "groundtruth" and veracity:
                node.attributes["veracity"] = veracity
            return node
        node = GraphNode(
            id=node_id,
            type="Time",
            name=value,
            confidence=confidence,
            source_agent=source_agent,
            source_round=source_round,
            provenance=provenance,
            text_span=None,
            last_updated=timestamp,
            attributes={"veracity": veracity} if veracity else {},
        )
        self.nodes[node.id] = node
        self.known_times.add(value)
        return node
    def _ensure_person_node(self, name: str) -> None:
        person_id = self._person_node_id(name)
        if person_id in self.nodes:
            node = self.nodes[person_id]
            if node.attributes is None:
                node.attributes = {}
            if node.attributes.get("veracity") != "groundtruth":
                node.attributes["veracity"] = "groundtruth"
            return
        node = GraphNode(
            id=person_id,
            type="Person",
            name=name,
            confidence=1.0,
            source_agent=name,
            source_round=0,
            provenance="role_card",
            text_span=None,
            last_updated=datetime.utcnow().isoformat() + "Z",
            attributes={"veracity": "groundtruth"},
        )
        self.nodes[person_id] = node

    def _person_node_id(self, name: str) -> str:
        return f"Person::{name}"

    def _location_node_id(self, name: str) -> str:
        return f"Location::{name}"

    def _object_node_id(self, name: str) -> str:
        return f"Object::{name}"

    def _time_node_id(self, value: str) -> str:
        return f"Time::{value}"

    def _next_id(self, node_type: str) -> str:
        self._id_counters[node_type] += 1
        prefix = node_type[0].upper()
        return f"{prefix}{self._id_counters[node_type]}"

    def _next_edge_id(self, edge_type: str) -> str:
        base = edge_type.upper()
        count = sum(1 for edge in self.edges.values() if edge.type == edge_type) + 1
        return f"{base}_{count}"

    def _split_response(self, text: str) -> (Optional[str], Optional[str]):
        thought_match = re.search(r"### THOUGHT:\s*(.*?)(?=### RESPONSE:|$)", text, re.S)
        response_match = re.search(r"### RESPONSE:\s*(.*)", text, re.S)
        thought = thought_match.group(1).strip() if thought_match else None
        response = response_match.group(1).strip() if response_match else None
        return thought, response

    def _collect_recent_facts(self) -> List[str]:
        window = self.fact_records[-5:]
        facts: List[str] = []
        for fact in reversed(window):
            parts: List[str] = [fact.speaker or "Unknown"]
            if fact.times:
                parts.append("@" + ",".join(sorted(fact.times)))
            if fact.locations:
                parts.append("在" + ",".join(sorted(fact.locations)))
            if fact.objects:
                parts.append("涉及" + ",".join(sorted(fact.objects)))
            snippet = self._truncate_text(fact.text, 120)
            if snippet:
                parts.append(snippet)
            if fact.veracity == "groundtruth":
                tag = "[solid]"
            elif fact.veracity == "claim":
                tag = "[claim]"
            else:
                tag = f"[{fact.veracity}]"
            facts.append(f"{tag} " + " ".join(parts))
        return facts

    def _collect_event_chain(self) -> List[str]:
        chain: List[str] = []
        event_times: List[Tuple[str, str]] = []
        for edge in self.edges.values():
            if edge.type == "happens_at":
                event_times.append((edge.source, edge.target))
        if not event_times:
            return chain
        for event_id, time_id in sorted(
            event_times,
            key=lambda item: (
                self._time_sort_key(self.nodes.get(item[1]).name if self.nodes.get(item[1]) else None),
                item[0],
            ),
        ):
            event_node = self.nodes.get(event_id)
            time_node = self.nodes.get(time_id)
            summary = None
            if event_node:
                summary = None
                if event_node.attributes:
                    summary = event_node.attributes.get("summary")
                summary = summary or event_node.text_span or event_node.name
            label = f"{event_node.name if event_node else event_id}({time_node.name if time_node else '未知时间'})"
            if summary:
                label += f": {self._truncate_text(summary, 100)}"
            chain.append(label)
        return chain

    def _summarize_dialogue(self) -> str:
        recent = self.statement_history[-6:]
        if not recent:
            return ""
        joined = "\n".join(
            f"R{item['round']}: {item['speaker']} -> {self._truncate_text(item['response'], 160)}"
            for item in recent if item.get("response")
        )
        limit = int(self.summary_config.get("max_tokens", 256) or 256) * 2
        return self._truncate_text(joined, limit)

    def _infer_provenance(self, template: str) -> str:
        if "clue" in template:
            return "clue"
        if "eval" in template:
            return "evaluation"
        if template.startswith("prompt_"):
            return "statement"
        return "system"

    def _edge_style(self, origin: str) -> Dict[str, str]:
        if origin == "clue" or origin == "groundtruth":
            return {"line_style": "solid", "veracity": "groundtruth"}
        if origin == "inference":
            return {"line_style": "solid", "veracity": "inference"}
        return {"line_style": "dashed", "veracity": "claim"}

    def _update_plan_for_round(self, round_index: int) -> None:
        topk = int(self.planner_config.get("topk", 2) or 1)
        plans: List[Dict[str, Any]] = self._build_conflict_plans(round_index, limit=topk)
        if len(plans) < topk:
            candidate_plan = self._build_statistical_plan(round_index)
            if candidate_plan:
                plans.append(candidate_plan)
        if not plans:
            plans.append(self._default_plan(round_index))
        plans = plans[: max(1, topk)]
        self.plan_log[round_index] = {
            "round": round_index,
            "plans": plans,
        }
        self.latest_plan = plans[0] if plans else None
        ordered_plans = [self.plan_log[k] for k in sorted(self.plan_log.keys())]
        self._write_json(self.plan_path, ordered_plans)

    def _build_conflict_plans(self, round_index: int, limit: int) -> List[Dict[str, Any]]:
        max_questions = int(self.planner_config.get("max_questions_per_round", limit) or limit)
        conflicts = [
            c for c in self.contradictions if c.get("status") == "open"
        ]
        conflicts.sort(key=lambda c: c.get("severity", 0), reverse=True)
        plans: List[Dict[str, Any]] = []
        for idx, conflict in enumerate(conflicts):
            if len(plans) >= limit or len(plans) >= max_questions:
                break
            severity = float(conflict.get("severity", 0.0))
            if severity < float(self.thresholds.get("conflict_severity", 0.6)):
                continue
            speakers = conflict.get("members", {}).get("speakers", [])
            query = self._build_conflict_query(conflict)
            plan = {
                "id": f"P{round_index}-{idx + 1}",
                "targets": speakers,
                "action": "Ask" if speakers else "Observe",
                "query": query,
                "expected_info_gain": round(severity, 3),
                "rationale": f"解决冲突 {conflict['id']}",
                "conflict_id": conflict["id"],
            }
            plans.append(plan)
        return plans

    def _build_conflict_query(self, conflict: Dict[str, Any]) -> str:
        members = conflict.get("members", {})
        times = members.get("times") or []
        locations = members.get("locations") or {}
        time_clause = "、".join(times) if times else "案发时段"
        location_clause = []
        for stmt_id, locs in locations.items():
            if not locs:
                continue
            location_clause.append(f"[{stmt_id}] {','.join(locs)}")
        location_text = "；".join(location_clause) if location_clause else "不同地点"
        return (
            f"围绕{time_clause}的行踪请逐分钟澄清，并解释为何存在 {location_text} 的冲突信息。"
        )

    def _build_statistical_plan(self, round_index: int) -> Optional[Dict[str, Any]]:
        if not self.candidate_cache:
            return None
        target_name, score = self._select_plan_target()
        if target_name is None:
            return None
        return {
            "id": f"P{round_index}-S",
            "targets": [target_name],
            "action": "Ask",
            "query": self._build_candidate_query(target_name),
            "expected_info_gain": round(max(score or 0.0, 0.0), 3),
            "rationale": "基于怀疑度差值生成的追问。",
        }

    def _default_plan(self, round_index: int) -> Dict[str, Any]:
        return {
            "id": f"P{round_index}-0",
            "targets": [],
            "action": "Observe",
            "query": "暂无高置信度冲突，继续收集基础事实。",
            "expected_info_gain": 0.0,
            "rationale": "No salient conflicts detected this round.",
        }

    def _select_plan_target(self) -> (Optional[str], Optional[float]):
        best_name = None
        best_score = None
        for name, stats in self.candidate_cache.items():
            query_prob = stats.get("query_probability")
            belief_prob = stats.get("belief_probability")
            if query_prob is None or belief_prob is None:
                continue
            score = query_prob - belief_prob
            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        if best_name is None and self.candidate_cache:
            sorted_candidates = sorted(
                self.candidate_cache.items(),
                key=lambda item: item[1].get("query", 0),
                reverse=True
            )
            if sorted_candidates:
                best_name = sorted_candidates[0][0]
                best_score = sorted_candidates[0][1].get("query", 0)
        return best_name, best_score

    def _build_candidate_query(self, target_name: Optional[str]) -> Optional[str]:
        if not target_name:
            return None
        return f"请详细复盘案发前后的关键行踪，特别说明是否曾进入核心案发地点。({target_name})"

    def _update_victory_metric(self, ranking: List[str], culprits: List[str]) -> None:
        best_mrr = 0.0
        for culprit in culprits:
            if culprit in ranking:
                rank_position = ranking.index(culprit)
                candidate_mrr = 1.0 / (rank_position + 1)
                if candidate_mrr > best_mrr:
                    best_mrr = candidate_mrr
        self.victory_mrr = best_mrr

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def _ensure_file(self, path: Path, default: Any) -> None:
        if not path.exists():
            self._write_json(path, default)

    def _ensure_debug_file(self) -> None:
        if not self.debug_trace_path.exists():
            self.debug_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.debug_trace_path, "w", encoding="utf-8") as f:
                f.write("")

    def _append_debug(self, message: str) -> None:
        with open(self.debug_trace_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        data = {
            "id": node.id,
            "type": node.type,
            "name": node.name,
            "confidence": node.confidence,
            "source_agent": node.source_agent,
            "source_round": node.source_round,
            "provenance": node.provenance,
            "text_span": node.text_span,
            "last_updated": node.last_updated,
        }
        if node.attributes:
            data["attributes"] = node.attributes
        return data

    def _edge_to_dict(self, edge: GraphEdge) -> Dict[str, Any]:
        data = {
            "id": edge.id,
            "type": edge.type,
            "source": edge.source,
            "target": edge.target,
            "confidence": edge.confidence,
            "source_agent": edge.source_agent,
            "source_round": edge.source_round,
            "provenance": edge.provenance,
            "text_span": edge.text_span,
            "last_updated": edge.last_updated,
        }
        if edge.attributes:
            data["attributes"] = edge.attributes
        return data

    def _edge_allowed(self, edge: GraphEdge) -> bool:
        if edge.type not in self.allowed_edge_types:
            return False
        source_node = self.nodes.get(edge.source)
        target_node = self.nodes.get(edge.target)
        if not source_node or not target_node:
            return False
        s_type = source_node.type
        t_type = target_node.type
        if edge.type == "was_at":
            return s_type == "Person" and t_type == "Location"
        if edge.type == "happens_at":
            return s_type == "Event" and t_type == "Time"
        if edge.type == "occurs_in":
            return s_type == "Event" and t_type == "Location"
        if edge.type == "involves":
            return s_type == "Event" and t_type in {"Person", "Object"}
        if edge.type == "used":
            return s_type == "Person" and t_type == "Object"
        if edge.type == "refers_to":
            return s_type == "Statement" and t_type in {"Person", "Object", "Event", "Clue"}
        if edge.type == "supports":
            return s_type in {"Statement", "Clue"} and t_type in {"Event", "Statement"}
        if edge.type == "contradicts":
            return s_type == "Statement" and t_type == "Statement"
        if edge.type == "about":
            return s_type == "Clue" and t_type in {"Object", "Event"}
        return False

    def _add_edge(self, edge: GraphEdge) -> GraphEdge:
        if not self._edge_allowed(edge):
            self._append_debug(
                f"Skip edge {edge.type}: {edge.source}->{edge.target} (types: "
                f"{self.nodes.get(edge.source).type if self.nodes.get(edge.source) else 'unknown'} -> "
                f"{self.nodes.get(edge.target).type if self.nodes.get(edge.target) else 'unknown'})"
            )
            return edge
        key = (edge.type, edge.source, edge.target)
        existing_id = self._edge_index.get(key)
        if existing_id:
            existing = self.edges[existing_id]
            if edge.confidence > existing.confidence:
                existing.confidence = edge.confidence
            existing.last_updated = edge.last_updated
            if edge.attributes:
                if existing.attributes is None:
                    existing.attributes = {}
                for key, value in edge.attributes.items():
                    if key == "line_style":
                        current = existing.attributes.get(key)
                        if current == "solid" and value != "solid":
                            continue
                        existing.attributes[key] = value
                        continue
                    if key == "veracity":
                        current = existing.attributes.get(key)
                        if current == "groundtruth" and value != "groundtruth":
                            continue
                        existing.attributes[key] = value
                        continue
                    existing.attributes[key] = value
            return existing
        self.edges[edge.id] = edge
        self._edge_index[key] = edge.id
        return edge

    def _truncate_text(self, text: Optional[str], limit: int) -> str:
        if not text:
            return ""
        text = text.strip()
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(limit - 1, 0)] + "…"

    def _time_sort_key(self, label: Optional[str]) -> float:
        if not label:
            return math.inf
        label = label.strip()
        match = re.search(r"(\d{1,2}):(\d{2})", label)
        if match:
            hour = int(match.group(1)) % 24
            minute = int(match.group(2)) % 60
            return hour * 60 + minute
        match = re.search(r"\b(1[0-2]|0?[1-9])(?:[:.](\d{2}))?\s*([ap]m)\b", label, re.I)
        if match:
            hour = int(match.group(1)) % 12
            if match.group(3).lower() == "pm":
                hour += 12
            minute = int(match.group(2)) if match.group(2) else 0
            return hour * 60 + minute
        match = re.search(r"(凌晨|清晨|上午|中午|下午|傍晚|晚上|夜里)?(\d{1,2})点(?:(\d{1,2})分)?", label)
        if match:
            period = match.group(1) or ""
            hour = int(match.group(2)) % 24
            minute = int(match.group(3)) if match.group(3) else 0
            if period in {"下午", "傍晚", "晚上", "夜里"} and hour < 12:
                hour += 12
            if period in {"凌晨"} and hour == 12:
                hour = 0
            if period in {"中午"} and hour < 12:
                hour = 12
            return hour * 60 + minute
        match = re.search(r"(\d{1,2})", label)
        if match:
            hour = int(match.group(1)) % 24
            return hour * 60
        return math.inf
