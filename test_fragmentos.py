import unittest

from langchain_core.documents import Document

from preparar_documentos import TOKENIZER, dividir_documento, preparar_documentos


class FragmentosTests(unittest.TestCase):
    def comprobar_fragmentos(self, texto):
        originales = TOKENIZER.encode_ordinary(texto)
        fragmentos = dividir_documento(Document(page_content=texto))
        reconstruido = bytearray()
        fin_anterior = 0
        for fragmento in fragmentos:
            inicio = fragmento.metadata["token_inicio"]
            fin = fragmento.metadata["token_fin"]
            self.assertLessEqual(fin - inicio, 500)
            self.assertEqual(
                fragmento.page_content,
                TOKENIZER.decode(originales[inicio:fin], errors="strict"),
            )
            if fin_anterior:
                self.assertEqual(fin_anterior - inicio, 50)
            for token in originales[fin_anterior:fin]:
                reconstruido.extend(TOKENIZER.decode_single_token_bytes(token))
            fin_anterior = fin
        self.assertEqual(reconstruido.decode("utf-8"), texto)

    def test_texto_largo_con_tildes_y_emojis(self):
        self.comprobar_fragmentos("Envíos rápidos 📦 a Córdoba y 日本. " * 200)

    def test_texto_corto_y_vacio(self):
        self.comprobar_fragmentos("Envíos a CABA.")
        self.comprobar_fragmentos("")

    def test_documento_del_tp(self):
        fragmentos = preparar_documentos()
        self.assertGreater(len(fragmentos), 1)
        self.assertTrue(all(f.metadata["source"] == "envios.md" for f in fragmentos))
        from preparar_documentos import DATA_DIR
        self.comprobar_fragmentos((DATA_DIR / "envios.md").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
