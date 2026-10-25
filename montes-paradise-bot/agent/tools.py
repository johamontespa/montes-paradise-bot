# agent/tools.py — Herramientas del agente
# Adaptado del kit WhatsApp AgentKit para usar RAG con ChromaDB en vez de busqueda
# por substring plano.

"""
Herramientas especificas del negocio de Montes Paradise.

- buscar_en_knowledge(): RAG real (ChromaDB + embeddings de Gemini). Es lo que
  agent/brain.py usa en cada turno para traer solo los fragmentos relevantes de
  knowledge/ en vez de mandar todo el contenido en el prompt.
- El resto son ganchos para acciones futuras (registrar lead, escalar a un
  humano) — hoy el agente solo informa y vende por texto; estas funciones quedan
  listas para conectarse a tool-use de Gemini cuando se necesite una accion real
  (ej. guardar un lead calificado en una hoja de calculo o notificar al dueño).
"""

import logging

import yaml

from agent.rag import buscar as rag_buscar

logger = logging.getLogger("agentkit")


def cargar_info_negocio() -> dict:
    """Carga la informacion del negocio desde config/business.yaml."""
    try:
        with open("config/business.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("config/business.yaml no encontrado")
        return {}


def obtener_horario() -> dict:
    """Retorna el horario de atencion del negocio."""
    info = cargar_info_negocio()
    return {
        "horario": info.get("negocio", {}).get("horario", "No disponible"),
    }


def buscar_en_knowledge(consulta: str, top_k: int = 4) -> str:
    """
    Busqueda semantica (RAG) sobre knowledge/*.md via ChromaDB.

    Retorna los fragmentos mas relevantes, formateados para pegarse directo en el
    contexto que se le manda a Gemini. Si la coleccion esta vacia o falla la busqueda,
    retorna un string vacio (brain.py debe manejar ese caso con gracia, sin inventar).
    """
    fragmentos = rag_buscar(consulta, top_k=top_k)
    if not fragmentos:
        return ""
    return "\n---\n".join(f"[{f['fuente']}]\n{f['texto']}" for f in fragmentos)


# ════════════════════════════════════════════════════════════
# Ganchos para cuando se conecte tool-use real (acciones, no solo texto):
#
# def registrar_lead(telefono: str, nombre: str, interes: str) -> None: ...
# def calificar_lead(telefono: str) -> str: ...  # "turista" | "kitesurfista" | "indefinido"
# def escalar_a_humano(telefono: str, motivo: str) -> None: ...
#   -> util para: precio no confirmado, queja, caso especial fuera del prompt
# ════════════════════════════════════════════════════════════
