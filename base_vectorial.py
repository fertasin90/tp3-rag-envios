from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "envios_horizonte"


def crear_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, request_timeout=30)


def abrir_coleccion():
    cliente = chromadb.PersistentClient(path=str(ROOT / ".chroma"))
    coleccion = cliente.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )
    if (coleccion.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("La base utiliza otro modelo de embeddings.")
    return coleccion
