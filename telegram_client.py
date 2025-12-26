# =====================================================================
# telegram_client.py — Envoi simple de messages Telegram
# =====================================================================
import os
import requests


def send_telegram_message(msg: str):
    """
    Envoie un message formaté Markdown à Telegram.
    Fonction synchrone simple → stable même en async loop.
    """

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram non configuré")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Telegram error:", e)
