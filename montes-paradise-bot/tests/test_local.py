# tests/test_local.py — Simulador de chat en terminal
# Adaptado del kit WhatsApp AgentKit

"""
Prueba tu agente sin necesitar WhatsApp ni Zernio/Meta.
Simula una conversacion en la terminal, con Gemini + RAG funcionando de verdad.

Antes de correr esto:
  1. pip install -r requirements.txt
  2. Llenar GEMINI_API_KEY en .env
  3. python scripts/build_knowledge_index.py   (indexa knowledge/ en ChromaDB)
  4. python tests/test_local.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.brain import generar_respuesta  # noqa: E402
from agent.memory import (  # noqa: E402
    guardar_mensaje,
    inicializar_db,
    limpiar_historial,
    obtener_historial,
)

TELEFONO_TEST = "test-local-001"


async def main():
    await inicializar_db()

    print()
    print("=" * 55)
    print("   Montes Paradise — Test Local del Agente")
    print("=" * 55)
    print()
    print("  Escribe mensajes como si fueras un cliente.")
    print("  Prueba con cosas como:")
    print("    'hola, cuantas personas caben?'")
    print("    'vamos a tirar kite en enero, que tal el viento?'")
    print("    'me parece caro'")
    print()
    print("  Comandos especiales:")
    print("    'limpiar'  — borra el historial")
    print("    'salir'    — termina el test")
    print()
    print("-" * 55)
    print()

    while True:
        try:
            mensaje = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nTest finalizado.")
            break

        if not mensaje:
            continue
        if mensaje.lower() == "salir":
            print("\nTest finalizado.")
            break
        if mensaje.lower() == "limpiar":
            await limpiar_historial(TELEFONO_TEST)
            print("[Historial borrado]\n")
            continue

        historial = await obtener_historial(TELEFONO_TEST)

        print("\nParaíso: ", end="", flush=True)
        respuesta, es_respuesta_real = await generar_respuesta(mensaje, historial)
        print(respuesta)
        print()

        if es_respuesta_real:
            await guardar_mensaje(TELEFONO_TEST, "user", mensaje)
            await guardar_mensaje(TELEFONO_TEST, "assistant", respuesta)


if __name__ == "__main__":
    asyncio.run(main())
