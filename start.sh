#!/usr/bin/env bash
# start.sh — Verifica el entorno y deja todo listo para correr el agente en local.
set -euo pipefail

echo "=== Montes Paradise — Setup del agente ==="
echo

# 1. Python 3.11+
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: no se encontro python3. Instala Python 3.11 o superior."
    exit 1
fi
PYVER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python detectado: $PYVER"

# 2. Entorno virtual
if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual (.venv)..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Dependencias
echo "Instalando dependencias..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 4. .env
if [ ! -f ".env" ]; then
    echo "Creando .env desde .env.example..."
    cp .env.example .env
    echo "  -> Falta llenar GEMINI_API_KEY y WHATSAPP_PROVIDER en .env"
fi

echo
echo "Listo. Proximos pasos:"
echo "  1. Llena GEMINI_API_KEY en .env (https://aistudio.google.com/apikey)"
echo "  2. python scripts/build_knowledge_index.py   # indexa knowledge/ en ChromaDB"
echo "  3. python tests/test_local.py                # probar el agente en terminal"
