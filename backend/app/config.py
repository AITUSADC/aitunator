"""Indexing settings only; chat, LLM and session configuration is not part of this package."""
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cs_students")
VECTOR_SIZE = 1024
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
MAX_CHARS_PER_CHUNK = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 16
CACHE_FILE = "runtime/points_cache.jsonl"
MANIFEST_FILE = "runtime/indexed_files.json"
KNOWLEDGE_BASE_DIR = "data/md"
PDF_SOURCE_DIR = "data/pdf"
PARSER_VERSION = "metadata-v3"
QDRANT_PAYLOAD_INDEXES = {
    key: "keyword" for key in ("program_code", "program_slug", "doc_type", "source_group",
    "source_format", "section_type", "language", "parent_id", "source", "updated_at")
}
QDRANT_PAYLOAD_INDEXES["priority"] = "integer"
