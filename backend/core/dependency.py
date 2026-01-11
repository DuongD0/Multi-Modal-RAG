from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from backend.core.config import Settings, settings
from backend.servies.model_service import ModelService, get_model_service
from backend.utils.json_docstore import JsonDocStore


@lru_cache
def get_settings() -> Settings:
    settings.ensure_dirs()
    return settings


@lru_cache
def get_model(cfg: Optional[Settings] = None) -> ModelService:
    cfg = cfg or get_settings()
    return get_model_service(cfg)


def _vector_dim(model_service: ModelService) -> int:
    sample = model_service.get_embedder().embed_query("dimension check")
    return len(sample)


def _build_new_store(cfg: Settings, model_service: ModelService) -> FAISS:
    dim = _vector_dim(model_service)
    index = faiss.IndexFlatIP(dim)
    return FAISS(
        embedding_function=model_service.get_embedder(),
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2=True,
    )


# Cache for vector store instance
_vector_store_cache: Optional[FAISS] = None


def get_vector_store(cfg: Optional[Settings] = None) -> FAISS:
    """Get or create the FAISS vector store.
    
    Loads from disk if exists, otherwise creates a new empty store.
    The store is cached for reuse across the application.
    """
    global _vector_store_cache
    if _vector_store_cache is not None:
        return _vector_store_cache
    
    cfg = cfg or get_settings()
    model_service = get_model(cfg)
    store_path: Path = cfg.vector_store_path

    if store_path.exists() and any(store_path.iterdir()):
        _vector_store_cache = FAISS.load_local(
            str(store_path),
            model_service.get_embedder(),
            allow_dangerous_deserialization=True,
        )
    else:
        _vector_store_cache = _build_new_store(cfg, model_service)
    
    return _vector_store_cache


def persist_vector_store(store: FAISS, cfg: Optional[Settings] = None) -> None:
    cfg = cfg or get_settings()
    cfg.ensure_dirs()
    store.save_local(str(cfg.vector_store_path))


@lru_cache
def get_docstore(cfg: Optional[Settings] = None) -> JsonDocStore:
    cfg = cfg or get_settings()
    return JsonDocStore(Path(cfg.docstore_path))

