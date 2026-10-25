# agent/brain.py — Cerebro del agente: conexion con Gemini + RAG
# Adaptado del kit WhatsApp AgentKit (que usa Claude) para usar Gemini (tier gratis)
# y para inyectar en cada turno los fragmentos de knowledge/ recuperados por RAG.

"""
Logica de IA del agente. Lee el system prompt de config/prompts.yaml, le agrega el
contexto recuperado por RAG (agent/rag.py -> ChromaDB) y genera la respuesta con la
API de Gemini.
"""

import logging
import os

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.tools import buscar_en_knowledge

load_dotenv()
logger = logging.getLogger("agentkit")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_client: "genai.Client | None" = None


def _cliente() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY no esta configurada en el .env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# El modelo se cambia desde .env, sin tocar el codigo.
#   gemini-2.5-flash       el balanceado (default) — tier gratis disponible
#   gemini-2.5-flash-lite  mas rapido y barato, menos capaz
#   gemini-2.5-pro         el mas capaz, tier gratis mas limitado
MODELO = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

# Tope de tokens de salida. WhatsApp son mensajes cortos.
MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS") or "1024")

# Cuantos fragmentos de knowledge/ trae el RAG por turno.
RAG_TOP_K = int(os.getenv("RAG_TOP_K") or "4")


def cargar_config_prompts() -> dict:
    """Lee toda la configuracion desde config/prompts.yaml."""
    try:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("config/prompts.yaml no encontrado")
        return {}


def cargar_system_prompt() -> str:
    """El system prompt base: quien es el agente, los avatares, la estrategia de ventas."""
    return cargar_config_prompts().get(
        "system_prompt", "Eres un asistente util. Responde siempre en espanol."
    )


def obtener_mensaje_error() -> str:
    return cargar_config_prompts().get(
        "error_message",
        "Lo siento, estoy teniendo problemas tecnicos. Por favor intenta de nuevo en unos minutos.",
    )


def obtener_mensaje_fallback() -> str:
    return cargar_config_prompts().get(
        "fallback_message", "Disculpa, no entendi tu mensaje. Podrias reformularlo?"
    )


def _construir_system_prompt_con_rag(mensaje: str) -> str:
    """
    Arma el system prompt final: el system_prompt fijo de prompts.yaml + SOLO los
    fragmentos de knowledge/ que el RAG considero relevantes para este mensaje.

    Esto es justo lo que hace que el prompt no crezca sin limite: en vez de mandar
    todo knowledge/ siempre, cada turno manda nada mas lo que hace falta para
    responder ESTE mensaje.
    """
    base = cargar_system_prompt()
    contexto_rag = buscar_en_knowledge(mensaje, top_k=RAG_TOP_K)

    if not contexto_rag:
        # Sin resultados de RAG (coleccion vacia o sin match): el agente sigue
        # funcionando con el system prompt base, pero sin inventar datos de knowledge/.
        return base

    return (
        f"{base}\n\n"
        "## Informacion relevante encontrada para este mensaje\n"
        "(viene de los archivos de /knowledge — usala para responder, y NO la "
        "compartas tal cual como si fuera una lista; integrala a tu respuesta en "
        "tono natural de WhatsApp)\n\n"
        f"{contexto_rag}"
    )


def _extraer_texto(respuesta) -> str:
    """Extrae el texto de la respuesta de Gemini."""
    texto = getattr(respuesta, "text", None)
    if texto:
        return texto.strip()

    # Fallback por si .text viene vacio pero hay candidates con partes de texto
    # (pasa, por ejemplo, si el modelo corto la respuesta por safety settings).
    try:
        partes = []
        for candidato in respuesta.candidates or []:
            for parte in candidato.content.parts or []:
                if getattr(parte, "text", None):
                    partes.append(parte.text)
        return "\n".join(partes).strip()
    except Exception:  # noqa: BLE001
        return ""


def _historial_a_contenidos(historial: list[dict]) -> list[types.Content]:
    """
    Convierte el historial [{"role": "user"|"assistant", "content": "..."}] al formato
    de Gemini, donde el rol del agente se llama "model" (no "assistant").
    """
    contenidos = []
    for m in historial:
        rol = "model" if m["role"] == "assistant" else "user"
        contenidos.append(types.Content(role=rol, parts=[types.Part(text=m["content"])]))
    return contenidos


async def generar_respuesta(mensaje: str, historial: list[dict]) -> tuple[str, bool]:
    """
    Genera una respuesta con Gemini, con contexto RAG inyectado para este turno.

    Args:
        mensaje: el mensaje nuevo del cliente
        historial: los mensajes anteriores, [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        (texto, es_respuesta_real) — "es_respuesta_real" es False cuando lo que se
        devuelve es un aviso tecnico (error o fallback) y no una respuesta real del
        agente. main.py usa esto para no guardar esos avisos en el historial.
    """
    if not mensaje or len(mensaje.strip()) < 2:
        return obtener_mensaje_fallback(), False

    system_prompt = _construir_system_prompt_con_rag(mensaje)
    contenidos = _historial_a_contenidos(historial)
    contenidos.append(types.Content(role="user", parts=[types.Part(text=mensaje)]))

    try:
        respuesta = await _cliente().aio.models.generate_content(
            model=MODELO,
            contents=contenidos,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_TOKENS,
            ),
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error llamando a Gemini: {e}")
        return obtener_mensaje_error(), False

    texto = _extraer_texto(respuesta)
    if not texto:
        logger.warning("Gemini devolvio una respuesta sin texto")
        return obtener_mensaje_fallback(), False

    try:
        uso = respuesta.usage_metadata
        logger.info(
            f"Respuesta generada con {MODELO} "
            f"({uso.prompt_token_count} in / {uso.candidates_token_count} out)"
        )
    except Exception:  # noqa: BLE001
        pass

    return texto, True
