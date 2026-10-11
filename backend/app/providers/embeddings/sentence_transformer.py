"""SentenceTransformer calls compatible with models without named prompts."""


def _prompt_options(model, role):
    return {"prompt_name": role} if role in getattr(model, "prompts", {}) else {}


def encode_query(model, text, **options):
    return model.encode(text, normalize_embeddings=True, **_prompt_options(model, "query"), **options)


def encode_documents(model, texts, **options):
    return model.encode(texts, normalize_embeddings=True, **_prompt_options(model, "document"), **options)
