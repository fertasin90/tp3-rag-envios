from pathlib import Path

import tiktoken
from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parent / "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def contar_tokens(texto: str) -> int:
    return len(TOKENIZER.encode_ordinary(texto))


def dividir_documento(documento: Document) -> list[Document]:
    tokens = TOKENIZER.encode_ordinary(documento.page_content)
    # Un token puede contener solo parte de una letra con tilde o de un emoji.
    limites = {0, len(tokens)}
    for posicion, token in enumerate(tokens):
        primer_byte = TOKENIZER.decode_single_token_bytes(token)[0]
        if primer_byte & 0b11000000 != 0b10000000:
            limites.add(posicion)

    fragmentos = []
    inicio = 0
    while inicio < len(tokens):
        fin = min(inicio + CHUNK_SIZE, len(tokens))
        if fin < len(tokens):
            # Ambos cortes deben caer entre caracteres completos.
            while fin not in limites or fin - CHUNK_OVERLAP not in limites:
                fin -= 1
            if fin - inicio <= CHUNK_OVERLAP:
                raise ValueError("No se encontró un corte seguro para este texto.")

        fragmentos.append(Document(
            page_content=TOKENIZER.decode(tokens[inicio:fin], errors="strict"),
            metadata={**documento.metadata, "token_inicio": inicio, "token_fin": fin},
        ))
        if fin == len(tokens):
            break
        inicio = fin - CHUNK_OVERLAP
    return fragmentos


def preparar_documentos() -> list[Document]:
    documentos = []
    for archivo in sorted(DATA_DIR.iterdir()):
        if archivo.is_file() and archivo.suffix.lower() in {".txt", ".md"}:
            texto = archivo.read_text(encoding="utf-8-sig").strip()
            if texto:
                documentos.append(
                    Document(page_content=texto, metadata={"source": archivo.name})
                )

    if not documentos:
        raise ValueError("La carpeta data necesita documentos .txt o .md con texto.")

    fragmentos = []
    for documento in documentos:
        fragmentos.extend(dividir_documento(documento))
    for numero, fragmento in enumerate(fragmentos, start=1):
        fragmento.metadata["chunk_id"] = numero
    return fragmentos


if __name__ == "__main__":
    fragmentos = preparar_documentos()
    print(f"Configuración: hasta {CHUNK_SIZE} tokens, overlap exacto {CHUNK_OVERLAP} del original.")
    print(f"Fragmentos obtenidos: {len(fragmentos)}")
    for fragmento in fragmentos:
        print(f"\nOrigen: {fragmento.metadata['source']}")
        print(f"Fragmento: {fragmento.metadata['chunk_id']}")
        print(f"Tokens: {contar_tokens(fragmento.page_content)}")
        print(f"Posiciones originales: {fragmento.metadata['token_inicio']} a {fragmento.metadata['token_fin']} (fin excluido)")
        print(fragmento.page_content)
