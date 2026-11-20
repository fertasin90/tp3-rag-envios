import unittest
from unittest.mock import AsyncMock, Mock, patch

from responder import RespuestaRAG, get_rag_response


class ResponderTests(unittest.IsolatedAsyncioTestCase):
    async def ejecutar_simulado(self, respuesta):
        coleccion = Mock()
        coleccion.count.return_value = 3
        coleccion.query.return_value = {
            "documents": [["Entrega desde el despacho en 2 a 4 días hábiles."]],
            "metadatas": [[{"source": "envios.md"}]],
        }
        embeddings = Mock()
        embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2])
        cadena = Mock()
        cadena.ainvoke = AsyncMock(return_value=respuesta)
        prompt = Mock()
        prompt.__or__ = Mock(return_value=cadena)
        with (
            patch("responder.load_dotenv"),
            patch("responder.abrir_coleccion", return_value=coleccion),
            patch("responder.crear_embeddings", return_value=embeddings),
            patch("responder.ChatOpenAI"),
            patch("responder.ChatPromptTemplate.from_messages", return_value=prompt),
        ):
            resultado = await get_rag_response("¿Cuánto tarda en llegar a CABA?")
        coleccion.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]], n_results=3,
            include=["documents", "metadatas"],
        )
        entrada = cadena.ainvoke.await_args.args[0]
        self.assertIn("envios.md", entrada["contexto"])
        self.assertIn("2 a 4", entrada["contexto"])
        return resultado

    async def test_fuente_recuperada(self):
        resultado = await self.ejecutar_simulado(RespuestaRAG(
            respuesta="Entre 2 y 4 días hábiles.", referencias=["envios.md"],
            informacion_suficiente=True,
        ))
        self.assertEqual(resultado.referencias, ["envios.md"])

    async def test_abstencion_sin_referencias(self):
        resultado = await self.ejecutar_simulado(RespuestaRAG(
            respuesta="Falta información sobre ese paquete.", referencias=["envios.md"],
            informacion_suficiente=False,
        ))
        self.assertEqual(resultado.referencias, [])

    async def test_rechaza_fuente_inventada(self):
        with self.assertRaises(ValueError):
            await self.ejecutar_simulado(RespuestaRAG(
                respuesta="Respuesta.", referencias=["inventado.md"],
                informacion_suficiente=True,
            ))

    async def test_rechaza_pregunta_vacia(self):
        with self.assertRaises(ValueError):
            await get_rag_response("   ")


if __name__ == "__main__":
    unittest.main()
