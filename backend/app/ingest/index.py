import json
import logging
import argparse
import hashlib
from importlib.util import find_spec
from pathlib import Path
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, OptimizersConfigDiff, PayloadSchemaType, PointStruct, VectorParams

from backend.app.config import (
    COLLECTION_NAME, VECTOR_SIZE, BATCH_SIZE, EMBEDDING_MODEL,
    CACHE_FILE, MANIFEST_FILE, KNOWLEDGE_BASE_DIR, PARSER_VERSION, PDF_SOURCE_DIR,
    QDRANT_PAYLOAD_INDEXES,
)
from backend.app.ingest.extract import build_chunks, build_markdown_chunks, parse_with_docling
from backend.app.rag.vector_store import create_qdrant_client
from backend.app.providers.embeddings.sentence_transformer import encode_documents

load_dotenv()

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.backends.mps.is_available():
        print("using MPS (apple)")
        return 'mps'
    if torch.cuda.is_available():
        print("using CUDA GPU")
        return 'cuda'
    print("using CPU")
    return 'cpu'


# utils

def make_id(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def make_chunk_id(payload: dict) -> str:
    normalized_text = " ".join(payload.get("text", "").split())
    text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
    key = "|".join([
        PARSER_VERSION,
        payload.get("source", ""),
        payload.get("parent_id", ""),
        str(payload.get("chunk_index", 0)),
        text_hash,
    ])
    return make_id(key)

def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def _load_manifest(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))

def _save_manifest(manifest: dict, path: str):
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def _encode_chunks_to_cache(
    chunks: list,
    model: SentenceTransformer,
    cache_path: str,
    description: str,
) -> None:
    cached_ids = set()
    if Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached_ids.add(json.loads(line)["id"])

    items_to_encode = [
        (text, make_chunk_id(payload), payload)
        for text, payload in chunks
        if make_chunk_id(payload) not in cached_ids
    ]

    if not items_to_encode:
        print("All chunks already in cache")
        return

    print(f"Encoding {len(items_to_encode)} new chunks...")
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("a", encoding="utf-8") as cache_f:
        for i in tqdm(range(0, len(items_to_encode), BATCH_SIZE), desc=description):
            batch = items_to_encode[i : i + BATCH_SIZE]
            texts = [item[0] for item in batch]
            vectors = encode_documents(
                model,
                texts,
                batch_size=BATCH_SIZE,
            ).tolist()
            for (text, point_id, payload), vector in zip(batch, vectors):
                cache_f.write(json.dumps({"id": point_id, "vector": vector, "payload": payload}, ensure_ascii=False) + "\n")

    print(f"Cache updated: {cache_path}")


def encode_markdown_and_cache(
    markdown_text: str,
    source_file: str,
    model: SentenceTransformer,
    cache_path: str,
) -> None:
    _encode_chunks_to_cache(
        build_markdown_chunks(markdown_text, source_file),
        model,
        cache_path,
        "BGE-M3 Markdown Encoding",
    )


def encode_pdf_and_cache(
    markdown_text: str,
    source_file: str,
    model: SentenceTransformer,
    cache_path: str,
) -> None:
    _encode_chunks_to_cache(
        build_chunks(
            markdown_text,
            source_file,
            source_format="pdf",
            source_group="raw",
            doc_type="program_descriptor",
        ),
        model,
        cache_path,
        "BGE-M3 PDF Encoding",
    )


# cache -> Qdrant

def upload_from_cache(client: QdrantClient, cache_path: str):
    if not Path(cache_path).exists():
        return

    existing_ids = set()
    try:
        offset = None
        while True:
            res, offset = client.scroll(collection_name=COLLECTION_NAME, limit=1000, offset=offset, with_payload=False)
            for p in res:
                existing_ids.add(str(p.id))
            if offset is None:
                break
    except Exception as e:
        print(f"Warning: could not fetch existing IDs from Qdrant: {e}")

    batch = []
    with open(cache_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['id'] in existing_ids:
                continue
            batch.append(PointStruct(id=row['id'], vector=row['vector'], payload=row['payload']))
            if len(batch) >= 100:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                batch = []
    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


# collections

def ensure_payload_indexes(client: QdrantClient) -> None:
    """Create Qdrant payload indexes for stable metadata filters when supported."""
    schema_by_name = {
        "keyword": PayloadSchemaType.KEYWORD,
        "integer": PayloadSchemaType.INTEGER,
    }
    for field_name, schema_name in QDRANT_PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_by_name[schema_name],
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "already has" not in message:
                print(f"Warning: could not create payload index {field_name}: {exc}")


def create_collection(client: QdrantClient):
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        size = info.config.params.vectors.size
        if size != VECTOR_SIZE:
            print(f"Recreating collection (dim = {size} -> {VECTOR_SIZE})")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection {COLLECTION_NAME} already exists")
            ensure_payload_indexes(client)
            return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000)
    )
    ensure_payload_indexes(client)
    print(f"Collection {COLLECTION_NAME} created")


# CLI

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Index curated Markdown and optional PDF documents into Qdrant")
    parser.add_argument("--index-md", type=str, metavar="MD_PATH", help="Path to a single curated Markdown file")
    parser.add_argument("--index-md-dir", type=str, metavar="DIR", default=None, help="Index all Markdown files in a directory")
    parser.add_argument("--index-pdf", type=str, metavar="PDF_PATH", help="Path to a single PDF file")
    parser.add_argument("--index-pdf-dir", type=str, metavar="DIR", help="Index all PDFs in a directory")
    parser.add_argument(
        "--index-all",
        action="store_true",
        help="Index curated Markdown from data/md/ and PDFs from data/pdf/ when present",
    )
    args = parser.parse_args()

    if not any([args.index_md, args.index_md_dir, args.index_pdf, args.index_pdf_dir, args.index_all]):
        parser.error("Provide --index-md, --index-md-dir, --index-pdf, --index-pdf-dir, or --index-all")

    pdf_requested = bool(args.index_pdf or args.index_pdf_dir)
    if args.index_all and Path(PDF_SOURCE_DIR).exists():
        pdf_requested = any(Path(PDF_SOURCE_DIR).rglob("*.pdf"))
    if pdf_requested and find_spec("docling") is None:
        parser.error("PDF indexing requires `uv sync --extra pdf`.")

    device = get_device()
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    client = create_qdrant_client()
    create_collection(client)
    manifest = _load_manifest(MANIFEST_FILE)

    def _index_md(md_path: Path) -> None:
        manifest_key = f"{PARSER_VERSION}:md:{md_path}"
        h = _file_md5(md_path)
        if manifest.get(manifest_key) == h:
            print(f"Skipping {md_path.name} (already indexed)")
            return
        print(f"Parsing curated Markdown {md_path.name}...")
        source_name = str(md_path.as_posix())
        encode_markdown_and_cache(
            md_path.read_text(encoding="utf-8"),
            source_name,
            embed_model,
            CACHE_FILE,
        )
        manifest[manifest_key] = h
        _save_manifest(manifest, MANIFEST_FILE)

    def _index_pdf(pdf_path: Path) -> None:
        manifest_key = f"{PARSER_VERSION}:pdf:{pdf_path}"
        h = _file_md5(pdf_path)
        if manifest.get(manifest_key) == h:
            print(f"Skipping {pdf_path.name} (already indexed)")
            return
        print(f"Parsing PDF {pdf_path.name}...")
        encode_pdf_and_cache(
            parse_with_docling(str(pdf_path)),
            pdf_path.name,
            embed_model,
            CACHE_FILE,
        )
        manifest[manifest_key] = h
        _save_manifest(manifest, MANIFEST_FILE)

    md_paths: list[Path] = []
    pdf_paths: list[Path] = []
    if args.index_all:
        curated_dir = Path(KNOWLEDGE_BASE_DIR)
        if curated_dir.exists():
            md_paths.extend(sorted(curated_dir.rglob("*.md")))
        pdf_dir = Path(PDF_SOURCE_DIR)
        if pdf_dir.exists():
            pdf_paths.extend(sorted(pdf_dir.rglob("*.pdf")))
    if args.index_md_dir:
        md_paths.extend(sorted(Path(args.index_md_dir).rglob("*.md")))
    if args.index_md:
        md_paths.append(Path(args.index_md))
    if args.index_pdf_dir:
        pdf_paths.extend(sorted(Path(args.index_pdf_dir).rglob("*.pdf")))
    if args.index_pdf:
        pdf_paths.append(Path(args.index_pdf))

    if md_paths:
        print(f"Found {len(md_paths)} curated Markdown files")
        for md_path in md_paths:
            _index_md(md_path)
    if pdf_paths:
        print(f"Found {len(pdf_paths)} PDF files")
        for pdf_path in pdf_paths:
            _index_pdf(pdf_path)

    upload_from_cache(client, CACHE_FILE)

    print("Done.")


if __name__ == "__main__":
    main()
