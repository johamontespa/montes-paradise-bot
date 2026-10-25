#!/usr/bin/env python3
# scripts/build_knowledge_index.py — Indexa knowledge/*.md en ChromaDB
#
# Correr cada vez que cambie algo en knowledge/ (nuevo precio, nueva politica, etc.):
#
#   python scripts/build_knowledge_index.py
#
# Requiere GEMINI_API_KEY configurada en .env (se usa para calcular los embeddings).

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s  %(message)s")

from agent.rag import reindexar_knowledge  # noqa: E402

if __name__ == "__main__":
    total = reindexar_knowledge()
    if total:
        print(f"\nListo: {total} fragmentos indexados en ChromaDB.")
    else:
        print("\nNo se indexo nada. Revisa que knowledge/*.md tenga contenido y que GEMINI_API_KEY este configurada.")
        sys.exit(1)
