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

