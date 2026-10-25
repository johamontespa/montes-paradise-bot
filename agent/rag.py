# agent/rag.py — Capa de RAG (Retrieval-Augmented Generation)
# Propio de este proyecto (Montes Paradise), no viene del kit original.

"""
Encapsula ChromaDB + embeddings de Gemini.

Usado por:
  - scripts/build_knowledge_index.py  -> indexa knowledge/*.md en Chroma
  - agent/tools.py / agent/brain.py   -> busca los fragmentos relevantes en cada turno

Por que un modulo aparte: tanto el script de indexacion como el agente en produccion
necesitan la MISMA logica de embeddings y la MISMA coleccion, para que lo que se
guarda y lo que se busca sean comparables.
"""

import logging
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger("agentkit")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001"
CHROMA_PATH = os.getenv("CHROMA_PATH") or "chroma_db"
COLLECTION_NAME = "montes_paradise_knowledge"
CARPETA_KNOWLEDGE = Path("knowledge")

TOP_K_DEFAULT = 4
# Cuantos caracteres por fragmento como maximo, para que cada chunk sea especifico
# (un fragmento muy largo diluye la busqueda semantica y gasta tokens de mas).
MAX_CHARS_POR_CHUNK = 1200

_client: "genai.Client | None" = None
_chroma_client = None
_collection = None


def _cliente_gemini() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY no esta configurada. Agregala al .env antes de indexar "
                "o de correr el agente."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _coleccion():
    """Coleccion persistente de Chroma en disco (CHROMA_PATH)."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # embedding_function=None: nosotros calculamos los embeddings a mano con Gemini
        # y se los pasamos ya listos a Chroma. Asi evitamos depender del wrapper de
        # embeddings que trae chromadb por defecto (que no es el de Gemini).
        _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _embed(textos: list[str], task_type: str) -> list[list[float]]:
    """
    Calcula embeddings con Gemini.

    task_type:
      "RETRIEVAL_DOCUMENT" -> para indexar contenido de knowledge/
      "RETRIEVAL_QUERY"    -> para la pregunta del cliente en cada turno
    """
    cliente = _cliente_gemini()
    respuesta = cliente.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=textos,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [list(e.values) for e in respuesta.embeddings]


def _chunkear_markdown(texto: str) -> list[str]:
    """
    Parte un archivo .md en fragmentos por seccion ("## "), y si una seccion queda
    muy larga, la vuelve a partir por parrafos.
    """
    secciones: list[str] = []
    actual: list[str] = []

    for linea in texto.splitlines():
        if linea.startswith("## ") and actual:
            secciones.append("\n".join(actual).strip())
            actual = [linea]
        else:
            actual.append(linea)
    if actual:
        secciones.append("\n".join(actual).strip())

    secciones = [s for s in secciones if s.strip()]

    fragmentos: list[str] = []
    for seccion in secciones:
        if len(seccion) <= MAX_CHARS_POR_CHUNK:
            fragmentos.append(seccion)
            continue
        # Seccion muy larga: la partimos por parrafos dobles, agrupando hasta el tope.
        parrafos = [p.strip() for p in seccion.split("\n\n") if p.strip()]
        buffer = ""
        for p in parrafos:
            if buffer and len(buffer) + len(p) + 2 > MAX_CHARS_POR_CHUNK:
                fragmentos.append(buffer)
                buffer = p
            else:
                buffer = f"{buffer}\n\n{p}" if buffer else p
        if buffer:
            fragmentos.append(buffer)

    return fragmentos or [texto.strip()]


def reindexar_knowledge() -> int:
    """
    Borra la coleccion y la vuelve a construir desde cero a partir de knowledge/*.md.
    Retorna cuantos fragmentos quedaron indexados.
    """
    if not CARPETA_KNOWLEDGE.is_dir():
        logger.error(f"No existe la carpeta {CARPETA_KNOWLEDGE}")
        return 0

    global _chroma_client, _collection
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 — no existia todavia, no pasa nada
        pass
    _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    documentos: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for ruta in sorted(CARPETA_KNOWLEDGE.glob("*.md")):
        texto = ruta.read_text(encoding="utf-8")
        fragmentos = _chunkear_markdown(texto)
        for i, frag in enumerate(fragmentos):
            documentos.append(frag)
            metadatas.append({"fuente": ruta.name})
            ids.append(f"{ruta.stem}-{i}")

    if not documentos:
        logger.warning("No se encontraron fragmentos para indexar en knowledge/")
        return 0

    embeddings = _embed(documentos, task_type="RETRIEVAL_DOCUMENT")
    _collection.add(ids=ids, documents=documentos, metadatas=metadatas, embeddings=embeddings)

    logger.info(f"Indexados {len(documentos)} fragmentos desde {CARPETA_KNOWLEDGE}/")
    return len(documentos)


def buscar(consulta: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    """
    Busca los fragmentos mas relevantes para la consulta.
    Retorna [{"texto": ..., "fuente": ...}, ...] ordenados por relevancia.
    """
    if not consulta or not consulta.strip():
        return []

    coleccion = _coleccion()
    if coleccion.count() == 0:
        logger.warning(
            "La coleccion de Chroma esta vacia. Corre "
            "'python scripts/build_knowledge_index.py' primero."
        )
        return []

    try:
        embedding_consulta = _embed([consulta], task_type="RETRIEVAL_QUERY")[0]
    except Exception as e:  # noqa: BLE001
        logger.error(f"No se pudo generar el embedding de la consulta: {e}")
        return []

    resultado = coleccion.query(
        query_embeddings=[embedding_consulta],
        n_results=min(top_k, coleccion.count()),
    )

    fragmentos = []
    documentos = (resultado.get("documents") or [[]])[0]
    metadatas = (resultado.get("metadatas") or [[]])[0]
    for doc, meta in zip(documentos, metadatas):
        fragmentos.append({"texto": doc, "fuente": (meta or {}).get("fuente", "?")})
    return fragmentos
