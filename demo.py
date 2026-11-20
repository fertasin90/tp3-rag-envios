import asyncio
import logging

from responder import get_rag_response

PREGUNTAS = [
    "¿Cuánto tarda un envío a CABA desde el despacho?",
    "¿Dónde está mi paquete 12345 y qué día exacto llegará?",
]


async def main() -> None:
    for pregunta in PREGUNTAS:
        print(f"\nPregunta: {pregunta}")
        resultado = await get_rag_response(pregunta)
        print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
