"""Language detection with deterministic fallback."""

from __future__ import annotations


def detect_language(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "en"
    if any("\u4e00" <= ch <= "\u9fff" for ch in sample):
        return "zh"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 42
        lang = detect(sample[:4000])
        if lang:
            return lang.split("-")[0].lower()
    except Exception:
        pass
    lower = sample.lower()
    if any(token in lower for token in ["bonjour", "merci", "veuillez", "remboursement"]):
        return "fr"
    if any(token in lower for token in ["hola", "gracias", "reembolso", "contraseña", "factura", "pago", "suscripcion", "suscripción", "necesito"]):
        return "es"
    if any(token in lower for token in ["hallo", "danke", "rechnung", "passwort"]):
        return "de"
    return "en"
