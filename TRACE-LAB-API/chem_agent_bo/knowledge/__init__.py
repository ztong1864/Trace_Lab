"""Knowledge retrieval module for Agentic BO."""

from chem_agent_bo.knowledge.provider import KnowledgeProvider
from chem_agent_bo.knowledge.query_builder import KnowledgeQueryBuilder
from chem_agent_bo.knowledge.node_query_builder import ReviewedKnowledgeQueryBuilder
from chem_agent_bo.knowledge.router import KnowledgeRouter
from chem_agent_bo.knowledge.rule_store import RuleStore
from chem_agent_bo.knowledge.schema import DocumentChunk, KnowledgeSnippet, KnowledgeUnit
from chem_agent_bo.knowledge.reviewed_schema import ReviewedKnowledgeEntry
from chem_agent_bo.knowledge.reviewed_experience import (
    ReviewedExperienceEntry,
    ReviewedExperienceStore,
)
from chem_agent_bo.knowledge.reviewed_store import ReviewedKnowledgeStore
from chem_agent_bo.knowledge.episodic_review import (
    EpisodicReviewCandidate,
    EpisodicReviewQueue,
    build_episodic_review_candidate,
)
from chem_agent_bo.knowledge.value_translation import (
    CACHE_PATH as VALUE_TRANSLATION_CACHE_PATH,
    load_translation_cache,
    merge_translation_entry,
    render_annotation_context,
    save_translation_cache,
)

try:
    from chem_agent_bo.knowledge.loader import (
        load_local_knowledge_into_store,
        load_rule_units_from_dir,
        load_units_from_local_dir,
    )
    from chem_agent_bo.knowledge.store import LanceDBStore
except ModuleNotFoundError:  # optional storage dependencies
    load_local_knowledge_into_store = None  # type: ignore[assignment]
    load_rule_units_from_dir = None  # type: ignore[assignment]
    load_units_from_local_dir = None  # type: ignore[assignment]
    LanceDBStore = None  # type: ignore[assignment]

try:
    from chem_agent_bo.knowledge.document_store import LanceDocumentStore, QdrantDocumentStore
    from chem_agent_bo.knowledge.hybrid_embeddings import QdrantHybridEmbedder
except ModuleNotFoundError:  # optional LangChain/Qdrant dependencies
    LanceDocumentStore = None  # type: ignore[assignment]
    QdrantDocumentStore = None  # type: ignore[assignment]
    QdrantHybridEmbedder = None  # type: ignore[assignment]

__all__ = [
    "DocumentChunk",
    "KnowledgeSnippet",
    "KnowledgeProvider",
    "KnowledgeRouter",
    "KnowledgeQueryBuilder",
    "ReviewedKnowledgeEntry",
    "ReviewedExperienceEntry",
    "ReviewedExperienceStore",
    "ReviewedKnowledgeStore",
    "ReviewedKnowledgeQueryBuilder",
    "EpisodicReviewCandidate",
    "EpisodicReviewQueue",
    "build_episodic_review_candidate",
    "LanceDocumentStore",
    "KnowledgeUnit",
    "LanceDBStore",
    "QdrantDocumentStore",
    "QdrantHybridEmbedder",
    "RuleStore",
    "VALUE_TRANSLATION_CACHE_PATH",
    "load_local_knowledge_into_store",
    "load_rule_units_from_dir",
    "load_translation_cache",
    "load_units_from_local_dir",
    "merge_translation_entry",
    "render_annotation_context",
    "save_translation_cache",
]
