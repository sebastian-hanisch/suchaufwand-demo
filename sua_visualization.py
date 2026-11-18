"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

from math import pi

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sua_constants as C


def _fest(fig: go.Figure, hoehe: int, **layout) -> go.Figure:
    fig.update_layout(height=hoehe, margin=dict(l=10, r=10, t=40, b=10), dragmode=False, **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _beschriftung(z, null_leer: bool = True) -> list[list[str]]:
    return [["" if (null_leer and v == 0) else f"{v:g}" for v in zeile] for zeile in np.asarray(z)]


def fig_karten(panels: list[dict], n: int, hoehe: int = 380) -> go.Figure:
    """Nebeneinander mehrere n x n-Karten. Je Panel: `werte` (Feld der Länge n*n), `titel`, `farbe` (Plotly-Farbskala), `text` (Beschriftung je Zelle, optional),
    `format` (Hover-Format). Zeile 0 oben, gleiche Maßstäbe, keine Farbleiste (der Titel nennt die Größe)."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p["titel"] for p in panels], horizontal_spacing=0.06)
    for i, p in enumerate(panels, start=1):
        w = np.asarray(p["werte"], dtype=float).reshape(n, n)
        text = p.get("text")
        fig.add_trace(go.Heatmap(z=w, text=text, texttemplate="%{text}" if text is not None else None, colorscale=p.get("farbe", C.COLOR_KARTE), showscale=False,
                                 hovertemplate=p.get("hover", "Zeile %{y}, Spalte %{x}: %{z:.3f}<extra></extra>"), xgap=1, ygap=1,
                                 textfont=dict(size=11 if n <= 8 else 8)), row=1, col=i)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{i if i > 1 else ''}", showticklabels=False, row=1, col=i)
        fig.update_xaxes(showticklabels=False, row=1, col=i)
    return _fest(fig, hoehe)


def fig_balken(werte: dict[str, float], namen: dict[str, str], farben: dict[str, str], titel: str, obergrenze: float | None = None, hoehe: int = 260) -> go.Figure:
    """Entdeckungswahrscheinlichkeit je Verfahren als waagrechte Balken (Prozent); optional die Obergrenze bei teilbarem Aufwand als gestrichelter Strich."""
    schl = list(werte)
    fig = go.Figure(go.Bar(y=[namen[k] for k in schl], x=[100 * werte[k] for k in schl], orientation="h", marker_color=[farben.get(k, C.COLOR_LINIE) for k in schl],
                           text=[f"{100 * werte[k]:.1f} %".replace(".", ",") for k in schl], textposition="auto", hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    if obergrenze is not None:
        fig.add_vline(x=100 * obergrenze, line_dash="dash", line_color="grey",
                      annotation_text="Obergrenze bei teilbarem Aufwand", annotation_position="bottom right", annotation_font_size=10)
    fig.update_xaxes(range=[0, 100.5], title="Entdeckungswahrscheinlichkeit in %")
    fig.update_yaxes(autorange="reversed")
    return _fest(fig, hoehe, title=dict(text=titel, font=dict(size=13)), showlegend=False)


def fig_weg(p, weg: list[int], n: int, titel: str, farbe: str = C.COLOR_WEG, hoehe: int = 400) -> go.Figure:
    """Prior als Karte, darüber der Weg des Teams (Start = Stern); Zahlen = Besuche je Zelle."""
    z = np.bincount(np.asarray(weg, dtype=int), minlength=n * n)
    fig = go.Figure(go.Heatmap(z=np.asarray(p).reshape(n, n), text=_beschriftung(z.reshape(n, n)), texttemplate="%{text}", colorscale=C.COLOR_KARTE, showscale=False,
                                hovertemplate="Zeile %{y}, Spalte %{x}: Prior %{z:.3f}<extra></extra>", xgap=1, ygap=1))
    ys, xs = [v // n for v in weg], [v % n for v in weg]
    rng_x = [x + 0.06 * ((i * 7) % 5 - 2) / 2 for i, x in enumerate(xs)]  # kleine Versetzung, damit Wiederholungen des Weges sichtbar bleiben
    rng_y = [y + 0.06 * ((i * 3) % 5 - 2) / 2 for i, y in enumerate(ys)]
    fig.add_trace(go.Scatter(x=rng_x, y=rng_y, mode="lines+markers", line=dict(color=farbe, width=2), marker=dict(size=4, color=farbe), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode="markers", marker=dict(symbol="star", size=14, color="gold", line=dict(color="black", width=1)),
                             hovertemplate="Basis (Schritt 1)<extra></extra>", showlegend=False))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return _fest(fig, hoehe, title=dict(text=titel, font=dict(size=13)))


def fig_wolke(pos: np.ndarray, S: float, mitten: list[tuple[float, float]], titel: str, hoehe: int = 420) -> go.Figure:
    """Driftwolke aus Teilchen mit dem Kreis um die Mitte, der 90 % einer Gauß-Wolke der Varianz S enthält."""
    fig = go.Figure(go.Scattergl(x=pos[:, 0], y=pos[:, 1], mode="markers", marker=dict(size=3, color=C.COLOR_LINIE, opacity=0.35), hoverinfo="skip", showlegend=False))
    r90 = float(np.sqrt(-2.0 * np.log(0.1) * S))
    th = np.linspace(0, 2 * pi, 100)
    for cx, cy in mitten:
        fig.add_trace(go.Scatter(x=cx + r90 * np.cos(th), y=cy + r90 * np.sin(th), mode="lines", line=dict(color=C.COLOR_WEG, width=2), hoverinfo="skip", showlegend=False))
    fig.update_yaxes(scaleanchor="x")
    return _fest(fig, hoehe, title=dict(text=titel, font=dict(size=13)))


def fig_flaeche(par: tuple, varianz, hoehe: int = 330) -> go.Figure:
    """Fläche des 90-%-Bereichs gegen die Zeit (doppelt logarithmisch, Bezug t = 1) mit den Steigungen 1 und 2 als Vergleich."""
    t = np.linspace(1.0, 8.0, 60)
    f = np.array([varianz(x, *par) / varianz(1.0, *par) for x in t])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=f, mode="lines", line=dict(color=C.COLOR_WEG, width=3), name="gewählte Werte"))
    fig.add_trace(go.Scatter(x=t, y=t, mode="lines", line=dict(color="grey", dash="dot"), name="Steigung 1 (wie die Zeit)"))
    fig.add_trace(go.Scatter(x=t, y=t ** 2, mode="lines", line=dict(color="grey", dash="dash"), name="Steigung 2 (Quadrat der Zeit)"))
    fig.update_xaxes(type="log", title="Zeit t")
    fig.update_yaxes(type="log", title="Fläche relativ zu t = 1")
    return _fest(fig, hoehe, legend=dict(orientation="h", y=-0.35))


def fig_verlust(kurve: list[tuple[float, float]], t0: float, verzoegerung: float, hoehe: int = 330) -> go.Figure:
    """Entdeckungswahrscheinlichkeit bei festem Aufwand gegen die Startzeit; markiert sind pünktlicher und verzögerter Start."""
    t = [a for a, _ in kurve]
    pv = [100 * b for _, b in kurve]
    fig = go.Figure(go.Scatter(x=t, y=pv, mode="lines+markers", line=dict(color=C.COLOR_LINIE, width=3), hovertemplate="Start bei t = %{x:.2f}: P = %{y:.1f} %<extra></extra>"))
    t1 = t0 * (1 + verzoegerung)
    fig.add_vline(x=t0, line_dash="dot", line_color="grey", annotation_text="pünktlich", annotation_position="top")
    fig.add_vline(x=t1, line_dash="dash", line_color=C.COLOR_WEG, annotation_text="verzögert", annotation_position="top")
    fig.update_xaxes(title="Startzeit der Suche t")
    fig.update_yaxes(title="Entdeckungswahrscheinlichkeit in %", range=[0, 100])
    return _fest(fig, hoehe, showlegend=False)
