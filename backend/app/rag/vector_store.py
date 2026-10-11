from qdrant_client import QdrantClient

from backend.app.config import QDRANT_API_KEY, QDRANT_URL


def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for local or authenticated remote deployments."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
