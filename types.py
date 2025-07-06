import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Optional


# ========= Ontology ==============

@dataclass(frozen=True)
class EntityType:
    label: str
    description: str


@dataclass(frozen=True)
class RelationType:
    predicate: str
    description: str
    bidirectional: bool = False


ENTITY_TYPES = [
    EntityType("USER", "The user"),
    EntityType("PERSON", "Person"),
    EntityType("ORG", "Organisation"),
    EntityType("ROLE", "Role / title"),
    EntityType("EVENT", "Event"),
    EntityType("LOCATION", "Location"),
    EntityType("DOCUMENT", "Document"),
    EntityType("CONTACT_POINT", "Contact point"),
    EntityType("DATETIME", "Date / time"),
    EntityType("ITEM", "Tangible item"),
    EntityType("ACTIVITY", "Activity"),
    EntityType("INTEREST", "Interest / hobby"),
    EntityType("MEDIA", "Media"),
    EntityType("MISC", "Miscellaneous"),
]


def format_entity_types() -> str:
    return "\n".join(f"{e.label}: {e.description}" for e in ENTITY_TYPES)


# TODO: refine
RELATION_TYPES = [
    RelationType("OWNS", "Owns (clear ownership)", bidirectional=False),
    # Social
    RelationType("FAMILY_MEMBER", "Interpersonal family relation", bidirectional=True),
    RelationType("FRIEND_OF", "Interpersonal friendship", bidirectional=True),
    RelationType("RELATIONSHIP_WITH", "General interpersonal relationship", bidirectional=True),
    RelationType("COMMUNICATES_WITH", "Communicates with", bidirectional=True),
    # Preferences
    RelationType("LIKES", "Likes / enjoys", bidirectional=False),
    RelationType("DISLIKES", "Dislikes / avoids", bidirectional=False),
    RelationType("INTERESTED_IN", "Demonstrable interest (interpersonal)", bidirectional=False),
    # Behavioural
    RelationType("PURCHASED", "Purchased", bidirectional=False),
    RelationType("PARTICIPATES_IN", "Participates in", bidirectional=False),
    # General
    RelationType("AFFILIATED_WITH", "Affiliated with", bidirectional=True),
    RelationType("SCHEDULED_FOR", "Scheduled for", bidirectional=False),
    RelationType("LOCATED_AT", "Located at", bidirectional=False),
    RelationType("RELATED_TO", "Related to", bidirectional=True),
]

PREDICATE_TO_BIDIRECTIONAL_LOOKUP = {
    r.predicate: r.bidirectional for r in RELATION_TYPES
}

def is_bidirectional(predicate: str) -> Optional[bool]:
    """ Returns whether the given relation predicate is known to be directional, defaults to False """
    return PREDICATE_TO_BIDIRECTIONAL_LOOKUP.get(predicate, False)


def format_relation_types() -> str:
    return "\n".join(f"{r.predicate}: {r.description}" for r in RELATION_TYPES)


# ========= Stage input / output representations ==============

@dataclass
class CleanChunk:
    """Container produced by the extraction stage"""

    source_file: str
    data: dict[str, Any]

    # TODO: special field for raw content, pass that through -> extraction
    dump: Optional[str] = None

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ChunkSummary:
    """Container produced by the summariser stage"""

    chunk: CleanChunk

    # TODO: encapsulate (prompt, raw_response, parsed_response)
    prompt: str
    raw_response: Optional[str] = None
    parsed_response: Optional[dict[str, Any]] = None

    summary_text: Optional[str] = None


@dataclass
class Entity:
    mention: str
    type: str  # e.g., PERSON, ORG, DATE
    reference: str  # Quote from the original doc
    document_id: Optional[str]

    def __str__(self):
        base = f"{self.mention} is a {self.type.lower()}."
        if self.reference:
            base += f" Context: {self.reference.lower()}."
        return base
    
    def get_id(self) -> str:
        return str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"ENTITY:{self.mention}|{self.type}")
        )[:4]


@dataclass
class Relation:
    subject: str  # e.g., "Alice"
    predicate: str  # e.g., "joined", "works_for"
    object: str  # e.g., "Apple"
    reference: str  # Quote from the original doc

    subject_id: Optional[str] = None  # maps to entity ID
    object_id: Optional[str] = None  # maps to entity ID

    def __str__(self):
        base = f"{self.subject} {self.predicate} {self.object}."
        if self.reference:
            base += f" Context: {self.reference.lower()}."
        return base

    def get_id(self) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"RELATION:{self.subject}|{self.predicate}|{self.object}",
            )
        )


@dataclass
class EntityContext:
    entity: Entity  # Retrieved entity
    score: float


@dataclass
class RelationContext:
    relation: Relation  # Retrieved relation
    score: float


@dataclass
class ExtractionResult:
    """Container produced by the entity + relation extraction stage"""

    chunk_summary: ChunkSummary
    prompt: str
    raw_response: Optional[str] = None
    parsed_response: Optional[dict[str, Any]] = None

    entities: Optional[list[Entity]] = None
    relations: Optional[list[Relation]] = None

    # Populated in linking step
    entity_kg_context: Optional[
        dict[str, list[EntityContext]]
    ] = None  # Entity ID -> [ (Entity, score) ]
    relation_kg_context: Optional[
        dict[str, list[RelationContext]]
    ] = None  # Relation ID -> [ (Relation, score) ]


# TODO: LinkedEntity and LinkedRelation can p be the same


@dataclass
class LinkedEntity:
    entity: Entity
    is_new: bool
    kg_id: Optional[str]
    chosen_score: Optional[float]
    reason: str


@dataclass
class LinkedRelation:
    relation: Relation
    is_new: bool
    kg_id: Optional[str]
    reason: str


@dataclass
class LLMCall:
    prompt: str
    raw_response: Optional[str] = None
    parsed_response: Optional[dict[str, Any]] = None


@dataclass
class LinkResult:
    """Container produced by the entity + relation extraction stage"""

    extraction_result: ExtractionResult

    # TODO: populate
    llm_calls: list[LLMCall] = field(default_factory=list)

    linked_entities: Optional[list[LinkedEntity]] = None
    linked_relations: Optional[list[LinkedRelation]] = None
