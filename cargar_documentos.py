import logging
from time import perf_counter

from dotenv import load_dotenv

from base_vectorial import ROOT, abrir_coleccion, crear_embeddings
from preparar_documentos import preparar_documentos

logger = logging.getLogger(__name__)


def cargar_documentos() -> None:
    inicio = perf_counter()
    fragmentos = preparar_documentos()
    textos = [fragmento.page_content for fragmento in fragmentos]
    metadatos = [fragmento.metadata for fragmento in fragmentos]
    ids = [f"{dato['source']}#{dato['chunk_id']}" for dato in metadatos]

    logger.info("Generando embeddings para %d fragmentos", len(textos))
    vectores = crear_embeddings().embed_documents(textos)
    coleccion = abrir_coleccion()
    coleccion.upsert(
        ids=ids, documents=textos, metadatas=metadatos, embeddings=vectores
    )
    # Quitamos fragmentos de versiones anteriores después de guardar la nueva carga.
    anteriores = set(coleccion.get(include=[])["ids"]) - set(ids)
    if anteriores:
        coleccion.delete(ids=sorted(anteriores))

    logger.info("Carga terminada en %.2f segundos", perf_counter() - inicio)
    print(f"Fragmentos guardados: {coleccion.count()}")
    print(f"Base persistente: {ROOT / '.chroma'}")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cargar_documentos()
