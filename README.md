# TP3 — RAG de Envíos Horizonte

Asistente sobre políticas de una empresa ficticia de envíos. Recupera fragmentos
de documentos en ChromaDB y genera una respuesta asíncrona validada con Pydantic,
con referencias a los archivos utilizados.

## Instalación en PowerShell

Requiere Python 3.10 o superior. Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Completá OPENAI_API_KEY en .env y guardá el archivo. OPENAI_MODEL permite elegir
un modelo compatible con salidas estructuradas estrictas. Las llamadas al proveedor
tienen costo. El archivo .env está excluido de Git.

## Ejecutar

```powershell
.\.venv\Scripts\python.exe preparar_documentos.py
.\.venv\Scripts\python.exe cargar_documentos.py
.\.venv\Scripts\python.exe demo.py
```

El primer programa permite inspeccionar los fragmentos. El segundo genera embeddings
y guarda la base persistente en .chroma. Volvé a ejecutarlo después de cambiar data.
La carga actualiza los fragmentos y elimina los que dejaron de existir.

El documento de ejemplo tiene 957 tokens. Se divide en ventanas de hasta 500 tokens,
con 50 tokens del original compartidos entre vecinos. Para conservar caracteres UTF-8
completos, los límites pueden retroceder ligeramente. El último fragmento puede ser
menor. Con el documento incluido se obtienen tamaños de 500, 500 y 57 tokens.
Interpretamos el tamaño de 500 de la consigna como tamaño de ventana y conservamos
el remanente final. La primera ejecución de tiktoken puede descargar su vocabulario.

Carga y búsqueda usan text-embedding-3-small. Se almacenan el texto, su vector y los
metadatos de origen. get_rag_response genera el embedding de la pregunta, recupera
hasta tres fragmentos y los envía como contexto al modelo. La llamada al modelo y
la generación del embedding de consulta son asíncronas; las operaciones síncronas
de ChromaDB se ejecutan en un hilo para evitar bloquear el bucle de eventos.

## Casos de la demo

- Pregunta sobre CABA desde el despacho. Se espera un plazo de 2 a 4 días hábiles,
  referencia envios.md e informacion_suficiente verdadero.
- Pregunta sobre la ubicación y fecha exacta del paquete 12345. Se espera reconocer
  que falta información, referencias vacías e informacion_suficiente falso.

La validación comprueba que las fuentes citadas figuren en los resultados recuperados.
La veracidad de la respuesta se evalúa además contra los documentos. El modo estricto
asegura la estructura de los campos; las pruebas simuladas verifican la lógica local.
La demo con el modelo real permite observar el comportamiento de generación.
Con solo tres fragmentos, la búsqueda actual recupera todo el documento. Para evaluar
la selección entre fuentes conviene ampliar el corpus y repetir las consultas.

## Pruebas sin API key

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Las pruebas de fragmentos verifican tamaños, solapamiento y reconstrucción con tildes
y emojis. Las de respuesta sustituyen ChromaDB y los clientes de modelos por dobles
locales para comprobar referencias, abstención y errores sin llamadas al proveedor.

## Archivos de entrega

- data/envios.md contiene las políticas ficticias.
- preparar_documentos.py lee y divide los documentos por tokens.
- base_vectorial.py centraliza la persistencia y el modelo de embeddings.
- cargar_documentos.py indexa los fragmentos.
- responder.py contiene el esquema Pydantic y get_rag_response.
- demo.py ejecuta los dos casos de ejemplo.
- test_fragmentos.py y test_responder.py permiten repetir las verificaciones locales.

Publicá el código, los documentos, requirements.txt, este README, .gitignore y
.env.example. El entorno, las credenciales y la base generada se excluyen de Git.
