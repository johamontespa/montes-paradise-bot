# agent/providers/zernio.py — Adaptador para Zernio
# Basado en la plantilla de WhatsApp AgentKit

"""
Zernio corre sobre la WhatsApp Cloud API de Meta: resuelve la conexion de la WhatsApp
Business Account, el inbox unificado y los webhooks firmados.

Documentacion: https://docs.zernio.com/platforms/whatsapp
"""

import hashlib
import hmac
import logging
import os

import httpx
from fastapi import Request

from agent.providers.base import MensajeEntrante, ProveedorWhatsApp

logger = logging.getLogger("agentkit")

BASE_URL_POR_DEFECTO = "https://zernio.com/api/v1"


class ProveedorZernio(ProveedorWhatsApp):
    """Proveedor de WhatsApp usando Zernio."""

    def __init__(self):
        self.api_key = os.getenv("ZERNIO_API_KEY", "")
        self.webhook_secret = os.getenv("ZERNIO_WEBHOOK_SECRET", "")
        self.account_id = os.getenv("ZERNIO_ACCOUNT_ID", "")
        self.base_url = (os.getenv("ZERNIO_BASE_URL") or BASE_URL_POR_DEFECTO).rstrip("/")

        if not self.api_key:
            logger.warning("ZERNIO_API_KEY no esta configurada: el agente no va a poder responder")
        if not self.webhook_secret:
            logger.warning(
                "ZERNIO_WEBHOOK_SECRET no esta configurado: los webhooks NO se verifican. "
                "Sirve para probar, pero no lo dejes asi en produccion."
            )

    async def verificar_firma(self, request: Request) -> bool:
        """Compara el header X-Zernio-Signature contra el HMAC-SHA256 del cuerpo crudo."""
        if not self.webhook_secret:
            return True

        firma_recibida = request.headers.get("X-Zernio-Signature") or request.headers.get(
            "X-Late-Signature"
        )
        if not firma_recibida:
            logger.warning("Llego un webhook sin firma X-Zernio-Signature: rechazado")
            return False

        cuerpo = await request.body()
        firma_esperada = hmac.new(
            self.webhook_secret.encode("utf-8"), cuerpo, hashlib.sha256
        ).hexdigest()

        try:
            iguales = hmac.compare_digest(firma_esperada, firma_recibida)
        except TypeError:
            logger.warning("La firma del webhook trae caracteres invalidos: rechazado")
            return False

        if not iguales:
            logger.warning("Firma de webhook invalida: rechazado")
            return False
        return True

    async def parsear_webhook(self, request: Request) -> list[MensajeEntrante]:
        """Normaliza el evento message.received de Zernio."""
        payload = await request.json()

        evento = payload.get("event")
        if evento != "message.received":
            logger.debug(f"Evento ignorado: {evento}")
            return []

        mensaje = payload.get("message") or {}
        if mensaje.get("platform") != "whatsapp":
            return []

        remitente = mensaje.get("sender") or {}
        telefono = (remitente.get("phoneNumber") or "").lstrip("+")
        if not telefono:
            telefono = remitente.get("businessScopedUserId") or remitente.get("id") or ""

        cuenta = payload.get("account") or {}

        return [
            MensajeEntrante(
                telefono=telefono,
                texto=mensaje.get("text") or "",
                mensaje_id=mensaje.get("platformMessageId") or mensaje.get("id") or "",
                es_propio=mensaje.get("direction") != "incoming",
                contexto={
                    "evento_id": payload.get("id", ""),
                    "conversation_id": mensaje.get("conversationId", ""),
                    "account_id": cuenta.get("id", ""),
                },
            )
        ]

    async def enviar_mensaje(
        self, telefono: str, mensaje: str, contexto: dict | None = None
    ) -> bool:
        """Responde dentro de la conversacion que abrio el cliente."""
        contexto = contexto or {}
        conversation_id = contexto.get("conversation_id")
        account_id = contexto.get("account_id") or self.account_id

        if not self.api_key:
            logger.error("No se puede enviar: falta ZERNIO_API_KEY")
            return False
        if not conversation_id or not account_id:
            logger.error(
                "No se puede enviar: el mensaje no trae conversation_id o account_id"
            )
            return False

        url = f"{self.base_url}/inbox/conversations/{conversation_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        evento_id = contexto.get("evento_id")
        if evento_id:
            headers["Idempotency-Key"] = f"agentkit-{evento_id}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as cliente:
                r = await cliente.post(
                    url,
                    json={"accountId": account_id, "message": mensaje},
                    headers=headers,
                )
        except httpx.HTTPError as e:
            logger.error(f"Error de red hablando con Zernio: {e}")
            return False

        if r.status_code == 200:
            return True

        detalle = r.text[:500]
        try:
            cuerpo = r.json()
            detalle = (
                f"{cuerpo.get('error')} "
                f"(type={cuerpo.get('type')}, code={cuerpo.get('code')})"
            )
        except ValueError:
            pass
        logger.error(f"Zernio rechazo el envio [{r.status_code}]: {detalle}")
        return False

    async def verificar_conexion(self) -> tuple[bool, str]:
        """Pregunta a Zernio el estado real del numero contra Meta."""
        if not self.api_key:
            return False, "Falta ZERNIO_API_KEY"
        if not self.account_id:
            return True, "ZERNIO_ACCOUNT_ID no configurado: se omite el chequeo del numero"

        try:
            async with httpx.AsyncClient(timeout=15.0) as cliente:
                r = await cliente.get(
                    f"{self.base_url}/whatsapp/number-info",
                    params={"accountId": self.account_id},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        except httpx.HTTPError as e:
            return False, f"No se pudo contactar a Zernio: {e}"

        if r.status_code != 200:
            return False, f"Zernio respondio {r.status_code}: {r.text[:200]}"

        telefono = r.json().get("phone") or {}
        return True, (
            f"Numero {telefono.get('display_phone_number', '?')} conectado "
            f"(calidad: {telefono.get('quality_rating', '?')})"
        )
