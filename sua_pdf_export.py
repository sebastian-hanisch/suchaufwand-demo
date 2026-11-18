"""PDF-Export der aktuellen Ansicht (fpdf2, eingebaute Schrift): Hinweis, Einstellungen, Kennzahlen und Verdikt als Text."""

import re

from fpdf import FPDF

_ERSATZ = {
    "α": "alpha", "β": "beta", "γ": "gamma", "κ": "kappa", "τ": "tau", "σ": "sigma", "π": "pi", "Δ": "Delta", "−": "-", "–": "-", "≈": "~", "≤": "<=", "≥": ">=",
    "→": "->", "„": '"', "“": '"', "”": '"', "’": "'", "₀": "0", "₁": "1", "₂": "2", "√": "Wurzel", "∞": "unendlich", "…": "...", " ": " ", " ": " ",
    "·": ".", "ₑ": "e", "ᵢ": "i",
}


def latin1(text: str) -> str:
    """Text für die eingebaute Schrift (Latin-1): Markdown-Zeichen entfernen, Sonderzeichen ersetzen."""
    text = re.sub(r"\*\*|`", "", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    for a, b in _ERSATZ.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def erzeuge_pdf(titel: str, hinweis: str, einstellungen: list[tuple[str, str]], kennzahlen: list[tuple[str, str]], verdikt: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, latin1(titel), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, latin1(hinweis), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    for kopf, zeilen in (("Einstellungen", einstellungen), ("Kennzahlen", kennzahlen)):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, kopf, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for name, wert in zeilen:
            pdf.multi_cell(0, 5.5, latin1(f"{name}: {wert}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikt", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, latin1(verdikt), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
