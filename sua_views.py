"""Die vier Ansichten der App (Streamlit): Regler im Hauptbereich beim Experiment, Ergebnis, Kernfrage, Expander.

Nur die gewählte Ansicht rechnet. Die Regler einer Ansicht bekommen ihre Standardwerte unmittelbar vor dem Zeichnen (`init_view_defaults`)."""

import numpy as np
import streamlit as st

import sua_constants as C
import sua_drift as D
import sua_evaluation as E
import sua_frozen as F
import sua_messung as K
import sua_pdf_export as PDF
import sua_presets as P
import sua_search as S
import sua_texts as T
import sua_visualization as V
from sua_evaluation import ivl, z


# ------------------------------------------------------------------------------------------------ Zwischenspeicher
@st.cache_data(show_spinner=False)
def _zuteilung(n, art, alpha, kappa, tau, seed):
    return E.analyse_zuteilung(n, art, alpha, kappa, tau, seed)


@st.cache_data(show_spinner=False)
def _gelaende(art, gamma, alpha, kappa, seed):
    return E.analyse_gelaende(art, gamma, alpha, kappa, seed)


@st.cache_data(show_spinner=False)
def _nichtfund(art, alpha, kappa, anteil, seed):
    return E.analyse_nichtfund(art, alpha, kappa, anteil, seed)


@st.cache_data(show_spinner=False)
def _falsches_alpha(art, alpha_ang, kappa, seed):
    return E.analyse_falsches_alpha(art, alpha_ang, kappa, seed)


@st.cache_data(show_spinner=False)
def _weg(art, alpha, T_, inst):
    return E.analyse_weg(art, alpha, T_, inst)


@st.cache_data(show_spinner=False)
def _drift(s0, dif, su, u, hyp, dv, t, t0, delay, L0):
    return E.analyse_drift(s0, dif, su, u, hyp, dv, t, t0, delay, L0)


@st.cache_data(show_spinner=False)
def _einheit():
    return D.einheitsnormale(C.TEILCHEN_ANZEIGE, 4711)


@st.cache_data(show_spinner=False)
def _kleine_instanzen():
    return K.kleine_rucksack_instanzen()


# ------------------------------------------------------------------------------------------------ Regler
def _zahl(x) -> str:
    return str(x).replace(".", ",") if isinstance(x, float) else str(x)


def regler_auswahl(key: str, label: str, help: str, namen: dict | None = None, radio: bool = False):
    """Auswahlregler (Schieberegler über feste Werte oder Umschalter) mit den Werten aus SETTING_SPECS."""
    sp = P.spec(key)
    fmt = (lambda v: namen[v]) if namen else _zahl
    if radio:
        return st.radio(label, sp.optionen, key=key, format_func=fmt, horizontal=True, help=help)
    return st.select_slider(label, options=sp.optionen, key=key, format_func=fmt, help=help)


def regler_zahl(key: str, label: str, help: str, disabled: bool = False):
    sp = P.spec(key)
    if sp.caster is float:
        return st.slider(label, float(sp.lo), float(sp.hi), key=key, step=float(sp.step), help=help, disabled=disabled)
    return st.number_input(label, int(sp.lo), int(sp.hi), key=key, step=int(sp.step), help=help, disabled=disabled)


def seed_regler(key: str, label: str = "Zufalls-Seed der Instanz"):
    c1, c2 = st.columns([3, 2])
    with c1:
        regler_zahl(key, label, "Bestimmt die erfundene Karte (und im Gelände die Kosten). Jeder Zufalls-Seed gibt dieselbe Instanz.")
    with c2:
        st.write("")
        st.button("🎲 Neue Instanz", key=f"{key}_neu", width="stretch", on_click=P.randomize_seed, args=(key,))


def pdf_knopf(key: str, titel: str, einstellungen, kennzahlen, verdikt: str, dateiname: str):
    st.download_button("📄 Ergebnis als PDF", data=PDF.erzeuge_pdf(titel, "Anwendungsbeispiel, keine Beratung. Alle Instanzen und Zahlen sind erfunden.", einstellungen, kennzahlen, verdikt),
                       file_name=dateiname, mime="application/pdf", key=key)


def verdikt_anzeigen(art_text: tuple[str, str]):
    art, text = art_text
    getattr(st, art)(text)


def punkte(luecke: float) -> str:
    """Delta einer Kennzahl: mein Wert minus Referenz (Optimum) in Prozentpunkten, negativ = schlechter."""
    return f"{-luecke:+.1f} Punkte".replace(".", ",") if abs(luecke) >= 0.05 else "±0 Punkte"


# ================================================================================================ Ansicht 1
def ansicht_zuteilung():
    P.init_view_defaults("zuteilung")
    st.markdown("## 🔬 Wer verliert gegen den Grenzzuwachs?")
    st.markdown(T.ANSICHT_EINLEITUNG["zuteilung"])
    c1, c2, c3 = st.columns(3)
    with c1:
        art = regler_auswahl("z_prior", "Prior (Karte)", "Ein Gipfel, zwei Gipfel (aus einer Driftwolke mit zwei Hypothesen) oder flach.", C.PRIOR_NAMEN, radio=True)
        n = regler_auswahl("z_gitter", "Gitter n×n", "Kantenlänge des Gitters.", {8: "8×8", 12: "12×12"}, radio=True)
    with c2:
        alpha = regler_auswahl("z_alpha", "Entdeckungsstärke α je Durchgang", "Ein Durchgang entdeckt das Ziel in seiner Zelle mit 1 − e^(−α).")
        kappa = regler_auswahl("z_kappa", "Budget (Durchgänge je Zelle)", "Budget B = Wert × Zahl der Zellen, in ganzen Durchgängen.")
    with c3:
        tau = regler_auswahl("z_tau", "Schwelle τ der „vollständigen“ Suche", "Eine Zelle gilt als vollständig abgesucht, wenn die Entdeckung in ihr mindestens τ erreicht.")
        seed_regler("z_seed")
    seed = int(st.session_state["z_seed"])
    a = _zuteilung(n, art, alpha, kappa, tau, seed)
    nmpf = {"opt": "Optimum (Grenzzuwachs)", "voll": f"„Vollständig“ (τ = {z(tau, 2)})", "durchgang": "Ein Durchgang je Zelle", "rechteck": "Bestes Rechteck"}
    cols = st.columns(4)
    for col, k in zip(cols, nmpf):
        col.metric(nmpf[k], f"{z(100 * a['P'][k])} %", delta="Referenz" if k == "opt" else punkte(a["luecke"][k]), delta_color="off" if k == "opt" else "normal")
    st.plotly_chart(V.fig_karten([
        {"werte": a["p"], "titel": "Prior p (Aufenthaltswahrscheinlichkeit)", "farbe": C.COLOR_KARTE},
        {"werte": a["z"]["opt"], "titel": f"Durchgänge des Optimums (B = {a['B']})", "farbe": "Greens", "text": V._beschriftung(a["z"]["opt"].reshape(n, n)),
         "hover": "Zeile %{y}, Spalte %{x}: %{z:.0f} Durchgänge<extra></extra>"}], n), width="stretch", key="chart_z_karten")
    st.plotly_chart(V.fig_balken(a["P"], nmpf, C.COLORS_VERFAHREN, f"Entdeckungswahrscheinlichkeit dieser Instanz (Zufalls-Seed {seed})", a["obergrenze"]), width="stretch", key="chart_z_balken")
    nie, masse = a["nie"]["opt"]
    st.caption(f"Einheit: ganze Durchgänge. Das Optimum lässt in dieser Instanz {z(100 * nie, 0)} % der Zellen unberührt (darin {z(100 * masse)} % der Prior-Masse); "
               "„ein Durchgang je Zelle“ lässt ab Budget N keine Zelle aus, das ist Absicht. Die gestrichelte Marke ist nur die Obergrenze bei teilbarem Aufwand.")
    kleinste, groesste = a["linie"]
    st.caption(f"Schwelle auf dem Grenzzuwachs: kleinster Grenzzuwachs eines genommenen Durchgangs {z(100 * kleinste, 3)} %, größter des nächsten {z(100 * groesste, 3)} % "
               f"({'die genommenen liegen darüber, wie es sein muss' if kleinste + 1e-12 >= groesste else 'Schwelle verletzt, das darf nicht vorkommen'}).")
    e1 = [("Prior", C.PRIOR_NAMEN[art]), ("Gitter", f"{n}x{n}"), ("alpha", z(alpha, 1)), ("Budget", f"{a['B']} Durchgaenge"), ("tau", z(tau, 2)), ("Zufalls-Seed", str(seed))]
    k1 = [(nmpf[k], f"{z(100 * a['P'][k])} %") for k in nmpf]
    row = F.zelle("s1", n, art, alpha, kappa)
    tau_row = F.zelle("tau", art, alpha, kappa, tau) if n == 8 else None
    v1 = E.verdict_zuteilung(n, art, alpha, kappa, tau, row, tau_row)
    pdf_knopf("pdf_z", "Suchaufwand: Zuteilung", e1, k1, v1[1], "suchaufwand_zuteilung.pdf")

    st.markdown("---")
    st.markdown("## 🔬 Gelände kostet")
    st.markdown("**Gelände:** Der Aufwand eines Durchgangs schwankt von Zelle zu Zelle (ganzzahlig, mindestens 1, im Mittel etwa 2; bei γ = 1,0 kosten einzelne Zellen deutlich mehr, in den Zufalls-Seeds 1 bis 30 bis zu 39); der Zusammenhang zwischen Gelände und Prior ist unabhängig erzeugt. "
                "Das Budget ist ein Anteil der Gesamtkosten. Verglichen wird mit dem **exakten** Optimum (Rucksack-Rechnung).")
    g1, g2, g3 = st.columns(3)
    with g1:
        gart = regler_auswahl("g_prior", "Prior (Karte)", "Ein oder zwei Gipfel.", C.PRIOR_NAMEN, radio=True)
        gamma = regler_auswahl("g_gamma", "Geländefaktor γ", "0 = überall gleich; größer = stärker schwankende Kosten (Variationskoeffizient im Mittel über die Zufalls-Seeds 1 bis 30: 0,53 bei 0,5 und 1,03 bei 1,0).")
    with g2:
        galpha = regler_auswahl("g_alpha", "Entdeckungsstärke α je Durchgang", "Stärke eines Durchgangs.")
        gkappa = regler_auswahl("g_kappa", "Budget (Anteil der Gesamtkosten)", "Budget = Wert × Summe der Durchgangskosten aller Zellen.")
    with g3:
        seed_regler("g_seed")
    gseed = int(st.session_state["g_seed"])
    g = _gelaende(gart, gamma, galpha, gkappa, gseed)
    gn = {"dp": "Exaktes Optimum (Rucksack)", "greedy_rat": "Greedy nach Zuwachs je Kosten", "voll_p": "„Vollständig“ nach Prior", "voll_rat": "„Vollständig“ nach Prior je Aufwand",
          "durch_p": "Ein Durchgang nach Prior", "durch_rat": "Ein Durchgang nach Prior je Aufwand", "rechteck": "Bestes Rechteck"}
    cols = st.columns(4)
    for col, k in zip(cols, ("dp", "greedy_rat", "voll_rat", "durch_rat")):
        col.metric(gn[k], f"{z(100 * g['P'][k])} %", delta="Referenz" if k == "dp" else punkte(g["luecke"][k]), delta_color="off" if k == "dp" else "normal")
    gz = g["z"]["greedy_rat"]
    st.plotly_chart(V.fig_karten([
        {"werte": g["p"], "titel": "Prior p", "farbe": C.COLOR_KARTE},
        {"werte": g["c"], "titel": "Kosten je Durchgang (Gelände)", "farbe": "Oranges", "text": V._beschriftung(g["c"].reshape(8, 8), False), "hover": "Zeile %{y}, Spalte %{x}: Kosten %{z:.0f}<extra></extra>"},
        {"werte": gz, "titel": f"Durchgänge des Greedy (B = {g['B']})", "farbe": "Greens", "text": V._beschriftung(gz.reshape(8, 8)), "hover": "Zeile %{y}, Spalte %{x}: %{z:.0f} Durchgänge<extra></extra>"}], 8, 330),
        width="stretch", key="chart_g_karten")
    st.plotly_chart(V.fig_balken({k: g["P"][k] for k in gn}, gn, {"dp": C.COLORS_VERFAHREN["opt"], "greedy_rat": "#2a9d8f", "voll_p": "#e08a1e", "voll_rat": "#f0b45f", "durch_p": "#8e5ea2",
                                                            "durch_rat": "#b48fc9", "rechteck": "#4c78a8"}, f"Entdeckungswahrscheinlichkeit dieser Instanz (Zufalls-Seed {gseed})", None, 330),
                    width="stretch", key="chart_g_balken")
    e2 = [("Prior", C.PRIOR_NAMEN[gart]), ("Gelaendefaktor gamma", z(gamma, 1)), ("alpha", z(galpha, 1)), ("Budget", f"{g['B']} Kosteneinheiten"), ("Zufalls-Seed", str(gseed))]
    k2 = [(gn[k], f"{z(100 * g['P'][k])} %") for k in gn]
    s2 = F.messreihe("s2")
    agg = E.aggregiere_s2(s2, gamma)
    v2 = E.verdict_gelaende(gamma, agg, F.messreihe("rucksack_klein") or None, F.zelle("s2", gart, gamma, galpha, gkappa))
    pdf_knopf("pdf_g", "Suchaufwand: Gelände", e2, k2, v2[1], "suchaufwand_gelaende.pdf")

    st.markdown("## ❓ Kernfrage: " + T.KERNFRAGEN["zuteilung"])
    st.markdown("**Zuteilung:**")
    verdikt_anzeigen(v1)
    s1 = F.messreihe("s1")
    if s1:
        verf = {"voll": "„Vollständig“ (τ = 0,95)", "durchgang": "Ein Durchgang je Zelle", "rechteck": "Bestes Rechteck"}
        zeilen = []
        for k, name in verf.items():
            r = {"Verfahren": name}
            for al in C.ALPHAS:
                ag = E.aggregiere_s1(s1, k, al)
                r[f"α = {_zahl(al)}"] = f"{z(ag['mittel'])} (max. {z(ag['maximum'])})" if ag else "–"
            zeilen.append(r)
        st.caption("Mittlere Lücke in Prozentpunkten je Entdeckungsstärke (Mittel über 20 Zellen: 8×8 und 12×12, ein und zwei Gipfel, fünf Budgets; in Klammern der größte Zellenmittelwert; je Zelle 30 Instanzen, vorgerechnet):")
        st.dataframe(zeilen, hide_index=True, width="stretch")
    st.markdown("**Gelände:**")
    verdikt_anzeigen(v2)

    with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
        tab_v, tab_e = st.tabs(["Verfahren", "🧮 Exakt (Aufzählung und Rucksack)"])
        with tab_v:
            st.markdown(T.METHODEN["zuteilung"])
        with tab_e:
            _exakt_zuteilung(alpha, seed, g, gamma)
    _grenzen_und_mathe("zuteilung")


def _exakt_zuteilung(alpha: float, seed: int, g: dict, gamma: float):
    st.markdown("**Aufzählung aller Zuteilungen.** Für eine kleine 4×4-Instanz (Prior „ein Gipfel“ aus dem Zufalls-Seed) mit Budget 5 werden alle 15 504 ganzzahligen Zuteilungen bewertet und mit dem Grenzzuwachs-Greedy verglichen.")
    if st.button("Alle Zuteilungen aufzählen (4×4, Budget 5)", key="exakt_aufzaehlen"):
        st.session_state["exakt_z"] = (alpha, seed)
    if st.session_state.get("exakt_z") == (alpha, seed):
        with st.spinner("Zähle alle Zuteilungen auf ..."):
            p = S.make_prior("ein_gipfel", 4, seed)
            zz = K.kompositionen(16, 5).astype(float)
            bester = float((p[None, :] * (1.0 - np.exp(-alpha * zz))).sum(axis=1).max())
            greedy = S.wert(p, alpha, S.greedy_zuteilung(p, alpha, 5))
        st.metric("Bester Wert aller Zuteilungen", f"{z(100 * bester, 3)} %", delta=f"{z(100 * (greedy - bester), 3)} Punkte für den Greedy (mein Wert minus bester)", delta_color="off")
    st.markdown("**Rucksack-Rechnung im Gelände.** Das exakte Optimum der Geländeinstanz oben: " + f"{z(100 * g['P']['dp'], 2)} %, der Greedy nach Zuwachs je Kosten {z(100 * g['P']['greedy_rat'], 2)} % (γ = {z(gamma, 1)}).")
    k = _kleine_instanzen()
    st.markdown(f"In {k['unter_optimum']} von {k['instanzen']} kleinen Instanzen (3×3, Kosten 1 bis 3, Budget 8) liegt der Greedy unter dem Optimum, größte Lücke {z(k['max_luecke'], 2)} Punkte.")


# ================================================================================================ Ansicht 2
def ansicht_nichtfund():
    P.init_view_defaults("nichtfund")
    st.markdown("## 🔬 Nichtfund aktualisiert die Karte")
    st.markdown(T.ANSICHT_EINLEITUNG["nichtfund"])
    st.warning("**Die Falle:** „Die Zelle ist nach Nichtfund noch wahrscheinlich, also sofort dort weitersuchen.“ Das Verfahren „dort weiter bis voll“ unten rechnet genau das; die Zahlen zeigen, was es kostet.")
    c1, c2, c3 = st.columns(3)
    with c1:
        art = regler_auswahl("n_prior", "Prior (Karte)", "Ein oder zwei Gipfel.", C.PRIOR_NAMEN, radio=True)
        verf = regler_auswahl("n_verfahren", "Runde 2 geht …", "Auf der aktualisierten Karte, auf der alten Karte (Zuteilung der Runde 1 wiederholt) oder „dort weiter bis voll“.", C.VERFAHREN_RUNDE2_NAMEN, radio=True)
    with c2:
        alpha = regler_auswahl("n_alpha", "Entdeckungsstärke α je Durchgang", "Stärke eines Durchgangs.")
        kappa = regler_auswahl("n_kappa", "Budget (Durchgänge je Zelle)", "Gesamtbudget beider Runden = Wert × 64 Zellen.")
    with c3:
        anteil = regler_auswahl("n_anteil", "Anteil des Budgets in Runde 1", "Die vorgerechneten Zahlen gelten für 50 %.", {0.25: "25 %", 0.5: "50 %", 0.75: "75 %"})
        seed_regler("n_seed")
    seed = int(st.session_state["n_seed"])
    r = _nichtfund(art, alpha, kappa, anteil, seed)
    n = r["n"]
    gesamt = {"aktualisiert": r["gesamt_a"], "alte_karte": r["gesamt_b"], "dort_weiter": r["gesamt_c"]}
    z2 = {"aktualisiert": r["z2a"], "alte_karte": r["z2b"], "dort_weiter": r["z2c"]}[verf]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entdeckt in Runde 1", f"{z(100 * r['d1'])} %", delta=f"{r['B1']} Durchgänge", delta_color="off")
    m2.metric(f"Gesamt, Runde 2 {C.VERFAHREN_RUNDE2_NAMEN[verf]}", f"{z(100 * gesamt[verf])} %", delta=punkte(S.luecke_punkte(r["gesamt_a"], gesamt[verf])) + " gegen aktualisiert", delta_color="normal")
    m3.metric("Gesamt, ein Vorgehen mit dem ganzen Budget", f"{z(100 * r['einplan'])} %", delta="Referenz", delta_color="off")
    m4.metric("Prüfzahl: aktualisiert − „alles auf einmal“", f"{z(100 * r['aktualisiert_minus_einplan'], 6)} Punkte", delta="soll 0 sein", delta_color="off")
    st.plotly_chart(V.fig_karten([
        {"werte": r["p"], "titel": f"Prior p und Runde 1 (Zahlen = Durchgänge, B₁ = {r['B1']})", "farbe": C.COLOR_KARTE, "text": V._beschriftung(r["z1"].reshape(n, n)),
         "hover": "Zeile %{y}, Spalte %{x}: Prior %{z:.3f}<extra></extra>"},
        {"werte": r["posterior"], "titel": f"Posterior nach Nichtfund und Runde 2 (Zahlen = Durchgänge, B₂ = {r['B2']})", "farbe": C.COLOR_KARTE, "text": V._beschriftung(z2.reshape(n, n)),
         "hover": "Zeile %{y}, Spalte %{x}: Posterior %{z:.3f}<extra></extra>"}], n), width="stretch", key="chart_n_karten")
    st.plotly_chart(V.fig_balken({k: gesamt[k] for k in C.VERFAHREN_RUNDE2}, {k: "Runde 2 " + C.VERFAHREN_RUNDE2_NAMEN[k] for k in C.VERFAHREN_RUNDE2}, C.COLORS_VERFAHREN,
                                 f"Gesamte Entdeckungswahrscheinlichkeit beider Runden (Zufalls-Seed {seed})", None, 220), width="stretch", key="chart_n_balken")
    p1 = r["p"]
    w = p1 * np.exp(-alpha * r["z1"])
    bes = r["z1"] > 0
    kleinste, groesste = r["linie1"]
    unbesucht_ueber = bool((~bes).any() and bes.any() and w[~bes].max() > w[bes].min() * (1 + 1e-12))
    quot = float(w[bes].max() / w[bes].min()) if bes.any() else 1.0
    st.caption(f"Nach Runde 1: kleinster Grenzzuwachs eines genommenen Durchgangs {z(100 * kleinste, 3)} % ≥ größter des nächsten {z(100 * groesste, 3)} % (Schwelle auf dem Grenzzuwachs). "
               f"Der Posterior der besuchten Zellen liegt im Band vom Faktor {z(quot, 2)} (höchstens e^α = {z(float(np.exp(alpha)), 2)}); "
               f"eine unbesuchte Zelle liegt {'über' if unbesucht_ueber else 'nicht über'} einer besuchten. „Alle abgesuchten Zellen gleich hoch“ gilt nur bei stetigem Aufwand.")
    e1 = [("Prior", C.PRIOR_NAMEN[art]), ("alpha", z(alpha, 1)), ("Budget", f"{r['B']} Durchgaenge"), ("Anteil Runde 1", f"{z(100 * anteil, 0)} %"), ("Runde 2", C.VERFAHREN_RUNDE2_NAMEN[verf]), ("Zufalls-Seed", str(seed))]
    k1 = [("Runde 1", f"{z(100 * r['d1'])} %")] + [("Gesamt " + C.VERFAHREN_RUNDE2_NAMEN[k], f"{z(100 * gesamt[k])} %") for k in gesamt]
    nf = F.messreihe("nichtfund")
    agg = E.aggregiere_nichtfund(nf)
    agg_rr = E.aggregiere_rechteck_runden(F.messreihe("rechteck_runden"))
    v1 = E.verdict_nichtfund(F.zelle("nichtfund", art, alpha, kappa), agg, agg_rr, anteil)
    pdf_knopf("pdf_n", "Suchaufwand: Nichtfund", e1, k1, v1[1], "suchaufwand_nichtfund.pdf")

    st.markdown("---")
    st.markdown("## 🔬 Falsch angenommenes α")
    st.markdown("Die Zuteilung wird mit einer **angenommenen** Entdeckungsstärke berechnet und mit der **wahren** bewertet. Der wahre Wert ist fest α = 1.")
    f1, f2, f3 = st.columns(3)
    with f1:
        fart = regler_auswahl("f_prior", "Prior (Karte)", "Ein oder zwei Gipfel.", C.PRIOR_NAMEN, radio=True)
    with f2:
        faa = regler_auswahl("f_alpha", "Angenommenes α", "Die wahre Stärke ist 1; 1 gibt keine Lücke.")
        fkappa = regler_auswahl("f_kappa", "Budget (Durchgänge je Zelle)", "Budget = Wert × 64 Zellen.")
    with f3:
        seed_regler("f_seed")
    fseed = int(st.session_state["f_seed"])
    f = _falsches_alpha(fart, faa, fkappa, fseed)
    m1, m2, m3 = st.columns(3)
    m1.metric("Zuteilung mit richtigem α = 1", f"{z(100 * f['P_richtig'])} %", delta="Referenz", delta_color="off")
    m2.metric(f"Zuteilung mit angenommenem α = {z(faa, 2)}", f"{z(100 * f['P_angenommen'])} %", delta=punkte(f["luecke"]), delta_color="normal")
    m3.metric("Lücke", f"{z(f['luecke'], 2)} Punkte", delta="Prozentpunkte Entdeckungswahrscheinlichkeit", delta_color="off")
    st.plotly_chart(V.fig_karten([
        {"werte": f["p"], "titel": "Prior p", "farbe": C.COLOR_KARTE},
        {"werte": f["z_richtig"], "titel": "Durchgänge mit richtigem α", "farbe": "Greens", "text": V._beschriftung(f["z_richtig"].reshape(8, 8)), "hover": "Zeile %{y}, Spalte %{x}: %{z:.0f} Durchgänge<extra></extra>"},
        {"werte": f["z_angenommen"], "titel": f"Durchgänge mit angenommenem α = {z(faa, 2)}", "farbe": "Purples", "text": V._beschriftung(f["z_angenommen"].reshape(8, 8)),
         "hover": "Zeile %{y}, Spalte %{x}: %{z:.0f} Durchgänge<extra></extra>"}], 8, 330), width="stretch", key="chart_f_karten")
    fa = F.messreihe("falsches_alpha")
    v2 = E.verdict_falsches_alpha(E.aggregiere_falsches_alpha(fa), F.zelle("falsches_alpha", fart, fkappa), faa)
    e2 = [("Prior", C.PRIOR_NAMEN[fart]), ("alpha angenommen", z(faa, 2)), ("alpha wahr", "1"), ("Budget", f"{f['B']} Durchgaenge"), ("Zufalls-Seed", str(fseed))]
    pdf_knopf("pdf_f", "Suchaufwand: falsches alpha", e2, [("Luecke", f"{z(f['luecke'], 2)} Punkte")], v2[1], "suchaufwand_falsches_alpha.pdf")

    st.markdown("## ❓ Kernfrage: " + T.KERNFRAGEN["nichtfund"])
    st.markdown("**Nichtfund:**")
    verdikt_anzeigen(v1)
    st.markdown("**Falsches α:**")
    verdikt_anzeigen(v2)
    with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
        tab_v, tab_e = st.tabs(["Verfahren", "🧮 Exakt (Rechteck-Muster, Aufzählung)"])
        with tab_v:
            st.markdown(T.METHODEN["nichtfund"])
        with tab_e:
            rr = F.messreihe("rechteck_runden")
            st.markdown("**Je Runde nur ein Rechteck** (8×8, zwei Runden, 30 Instanzen je Zelle, vorgerechnet): Lücke zum frei verteilten Optimum in Prozentpunkten [Intervall]. "
                        "„Kurzsichtig“ = je Runde das beste Rechteck auf der aktualisierten Karte; „Vorausschau“ = unter den 40 besten Rechtecken der Runde 1 jeweils das beste der Runde 2 (untere Schranke für das beste Rechteckpaar); „ein Rechteck“ = das ganze Budget in einem Rechteck.")
            zeilen = [{"Prior": C.PRIOR_NAMEN[k.split("|")[0]], "α": _zahl(float(k.split("|")[1])), "Budget": k.split("|")[2], "Kurzsichtig": ivl(v["kurzsichtig"], 2),
                       "Vorausschau": ivl(v["vorausschau"], 2), "Ein Rechteck": ivl(v["ein_rechteck"], 2), "Vorausschau besser (Anteil)": z(v["vorteil_anteil"], 2)} for k, v in rr.items()]
            st.dataframe(zeilen, hide_index=True, width="stretch")
    _grenzen_und_mathe("nichtfund")


# ================================================================================================ Ansicht 3
def ansicht_weg():
    P.init_view_defaults("weg")
    st.markdown("## 🔬 Was kostet der Weg?")
    st.markdown(T.ANSICHT_EINLEITUNG["weg"])
    c1, c2, c3 = st.columns(3)
    with c1:
        art = regler_auswahl("w_prior", "Prior (Karte)", "Ein Gipfel, zwei Gipfel oder flach (alle Wege über T verschiedene Zellen sind dann optimal).", C.PRIOR_NAMEN, radio=True)
        alpha = regler_auswahl("w_alpha", "Entdeckungsstärke α je Durchgang", "Bei großem α zählt jede Zelle praktisch nur einmal: in der Wertung ein Orienteering-Problem.")
    with c2:
        Tn = regler_auswahl("w_T", "Zeitschritte T des Teams", "Ein Schritt = ein Durchgang an der aktuellen Zelle; die Anfahrt zählt als Suchen.")
        inst = regler_zahl("w_inst", "Instanz Nr. (Zufalls-Seed 1 bis 30)", "Die 30 Instanzen, über die gemittelt wurde; nur für diese liegen die vorgerechneten Wege vor.")
    with c3:
        verf = regler_auswahl("w_verfahren", "Weg auf der Karte", "Welcher Weg gezeichnet wird.", C.VERFAHREN_WEG_NAMEN)
    a = _weg(art, alpha, Tn, int(inst))
    if verf not in a["wege"]:
        st.warning("Für diesen Weg liegt keine vorgerechnete Lösung vor; gezeigt wird der Weg „Greedy mit Ziel“.")
        verf = "greedy_ziel"
    bew = a["bew"].get(verf)
    zusatz = "" if bew in (None, True) else " (beste gefundene Lösung, nicht bewiesen)"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(C.VERFAHREN_WEG_NAMEN[verf], f"{z(100 * a['P'][verf])} %", delta=("Referenz" if verf == "exakt" else punkte(a["luecke"].get(verf, 0.0))), delta_color="off" if verf == "exakt" else "normal")
    m2.metric("Wegoptimum (beste bekannte Lösung)", f"{z(100 * a['beste'])} %", delta="Referenz", delta_color="off")
    m3.metric("Optimum ohne Weg (T Durchgänge frei)", f"{z(100 * a['schranke'])} %", delta=f"Wegpreis {z(a['wegpreis'], 2)} Punkte", delta_color="off")
    m4.metric("Überschätzung der Orienteering-Fassung", f"{z(a['ueberschaetzung'])} Punkte" if a["ueberschaetzung"] is not None else "–", delta="Prämien-Summe minus wahrer Wert", delta_color="off")
    st.plotly_chart(V.fig_weg(a["p"], a["wege"][verf], a["n"], f"{C.VERFAHREN_WEG_NAMEN[verf]}{zusatz}: Karte p, Weg ab der Basis (Stern), Zahlen = Besuche je Zelle"), width="stretch", key="chart_w_weg")
    zeilen = []
    for k, name in C.VERFAHREN_WEG_NAMEN.items():
        if k in a["P"]:
            zeilen.append({"Weg": name, "Entdeckungswahrscheinlichkeit (%)": z(100 * a["P"][k], 2), "Lücke zum Wegoptimum (Punkte)": z(100 * (a["beste"] - a["P"][k]), 2) if k != "exakt" else "–",
                           "Zellen besucht": int(len(set(a["wege"][k])))})
    st.dataframe(zeilen, hide_index=True, width="stretch")
    st.caption("Zahlen dieser einen Instanz. Mittel über 30 Instanzen mit Intervall stehen im Verdikt unten (vorgerechnet).")
    tab = _weg_tabelle(art, alpha, Tn)
    v = E.verdict_weg(art, alpha, Tn, tab, a)
    e = [("Prior", C.PRIOR_NAMEN[art]), ("alpha", z(alpha, 1)), ("T", str(Tn)), ("Instanz", str(inst)), ("Weg", C.VERFAHREN_WEG_NAMEN[verf])]
    pdf_knopf("pdf_w", "Suchaufwand: Weg", e, [(k, f"{z(100 * a['P'][k])} %") for k in a["P"]], v[1], "suchaufwand_weg.pdf")
    st.markdown("## ❓ Kernfrage: " + T.KERNFRAGEN["weg"])
    verdikt_anzeigen(v)
    with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
        tab_v, tab_e = st.tabs(["Verfahren", "🧮 Exakt (CP-SAT, vorgerechnet)"])
        with tab_v:
            st.markdown(T.METHODEN["weg"])
        with tab_e:
            _exakt_weg()
    _grenzen_und_mathe("weg")


def _weg_tabelle(art: str, alpha: float, Tn: int):
    return F.weg_tabelle(5, art, alpha, Tn) if art != "flach" else None


def _exakt_weg():
    st.markdown("**CP-SAT, vorgerechnet.** Das Wegoptimum und die Orienteering-Fassung wurden mit `tools/sua_make_weg.py` gelöst und in `data/sua_weg.json` eingefroren; die App rechnet CP-SAT nie live. "
                "Instanzen ohne Beweis sind markiert („beste gefundene Lösung“); dort sind die Lücken der Verfahren eher zu klein.")
    d = F.laden("sua_weg")
    if not d or "g8" not in d:
        st.info("Die Tabelle für 8×8 liegt nicht vor.")
        return
    zeilen = []
    for key, t in d["g8"]["tabelle"].items():
        art, al, Tn = key.split("|")
        zeilen.append({"Prior": C.PRIOR_NAMEN[art], "α": _zahl(float(al)), "T": Tn, "Wegoptimum %": z(t["opt_mittel"]), "Wegpreis": ivl(t["wegpreis"]), "Greedy kurzsichtig": ivl(t["greedy_kurz"]),
                       "Greedy mit Ziel": ivl(t["greedy_ziel"]), "Mähen ganz": ivl(t["maehen_ganz"]), "Mähen im Rechteck": ivl(t["maehen_rechteck"]), "nicht bewiesen": t["unbewiesen"]})
    st.markdown("**8×8 (nur Tabelle, 30 Instanzen je Zelle)**: Lücke zum besten bekannten Weg in Prozentpunkten [Intervall].")
    st.dataframe(zeilen, hide_index=True, width="stretch")


# ================================================================================================ Ansicht 4
def ansicht_drift():
    P.init_view_defaults("drift")
    st.markdown("## 🔬 Wie schnell wächst die Wolke, und was kostet späteres Starten?")
    st.markdown(T.ANSICHT_EINLEITUNG["drift"])
    c1, c2, c3 = st.columns(3)
    with c1:
        s0 = regler_zahl("d_s0", "Startunsicherheit s₀", "Streuung der Startposition je Achse.")
        dif = regler_zahl("d_D", "Diffusion D", "Zufällige Bewegung: geht mit 2·D·t in die Varianz ein.")
        su = regler_zahl("d_sU", "Geschwindigkeitsunsicherheit sU", "Je Teilchen konstante Unsicherheit der Geschwindigkeit: geht mit sU²·t² in die Varianz ein.")
        u = regler_zahl("d_u", "Mittlere Driftgeschwindigkeit Ū", "Verschiebt die Wolke nach rechts; ändert Fläche und Entdeckung nicht.")
    with c2:
        hyp = regler_auswahl("d_hyp", "Zahl der Hypothesen", "Zwei Hypothesen: Querwind links und rechts, die Wolke teilt sich in zwei Hälften.", radio=True)
        dv = regler_zahl("d_dv", "Abstand dv der Hypothesen", "Geschwindigkeitsunterschied quer zum Wind; wirkt nur bei zwei Hypothesen.", disabled=(hyp == 1))
        t = regler_zahl("d_t", "Zeit t der gezeigten Wolke", "Zeitpunkt der Teilchenwolke im Bild.")
    with c3:
        t0 = regler_zahl("d_t0", "Startzeit der Suche t₀", "Zeitpunkt, zu dem die Suche pünktlich beginnt.")
        delay = regler_auswahl("d_delay", "Verzögerung (Anteil von t₀)", "Wie viel später die Suche beginnt.", {0.10: "10 %", 0.25: "25 %", 0.50: "50 %", 1.00: "100 %"})
        L0 = regler_auswahl("d_L0", "Aufwand L₀ bei pünktlichem Start", "Maß für den Aufwand: L = √(αE/(πS)); größer heißt reichlicher Aufwand.")
    a = _drift(s0, dif, su, u, hyp, dv, t, t0, delay, L0)
    sz = F.drift_szenarien()
    v = E.verdict_drift(a, sz)
    if a is None:
        verdikt_anzeigen(v)
        _grenzen_und_mathe("drift")
        return
    par = a["par"]
    S_t = D.varianz(t, *par)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Varianz S(t) je Achse", z(S_t, 2), delta=f"Fläche {z(D.flaechenverhaeltnis(t, par), 2)}× so groß wie bei t = 1", delta_color="off")
    m2.metric("P bei pünktlichem Start", f"{z(100 * a['p0'])} %", delta="Referenz", delta_color="off")
    m3.metric("P bei verzögertem Start", f"{z(100 * a['p_verzoegert'])} %", delta=f"{-100 * a['verlust']:+.1f} % von P".replace(".", ","), delta_color="normal")
    m4.metric("Elastizität des Verlusts", z(a["elastizitaet"], 2), delta="unter 1: unterproportional", delta_color="off")
    pos = D.wolke(par, t, u, hyp, dv, _einheit())
    mitten = [(u * t, dv * t), (u * t, -dv * t)] if hyp == 2 else [(u * t, 0.0)]
    st.plotly_chart(V.fig_wolke(pos, S_t, mitten, f"Wolke bei t = {z(t, 1)}: {C.TEILCHEN_ANZEIGE} Teilchen, Kreise = Bereich mit 90 % einer Gauß-Wolke"), width="stretch", key="chart_d_wolke")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(V.fig_flaeche(par, D.varianz), width="stretch", key="chart_d_flaeche")
    with g2:
        st.plotly_chart(V.fig_verlust(a["kurve"], t0, delay), width="stretch", key="chart_d_verlust")
    st.caption("Links: Fläche des 90-%-Bereichs gegen die Zeit, doppelt logarithmisch (Steigung 1 = wächst wie die Zeit, Steigung 2 = wie das Quadrat der Zeit). Rechts: Entdeckungswahrscheinlichkeit bei festem Aufwand und späterem Start.")
    e = [("s0", z(s0, 1)), ("D", z(dif, 2)), ("sU", z(su, 2)), ("Hypothesen", str(hyp)), ("t0", z(t0, 1)), ("Verzoegerung", f"{z(100 * delay, 0)} %"), ("L0", z(L0, 1))]
    k = [("Varianz S(t)", z(S_t, 2)), ("P puenktlich", f"{z(100 * a['p0'])} %"), ("P verzoegert", f"{z(100 * a['p_verzoegert'])} %"), ("Verlust", f"{z(100 * a['verlust'])} %"), ("Elastizitaet", z(a["elastizitaet"], 2))]
    pdf_knopf("pdf_d", "Suchaufwand: Drift und Verzögerung", e, k, v[1], "suchaufwand_drift.pdf")
    st.markdown("## ❓ Kernfrage: " + T.KERNFRAGEN["drift"])
    verdikt_anzeigen(v)
    verdikt_anzeigen(E.verdict_elastizitaet(a))
    with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
        tab_v, tab_e = st.tabs(["Verfahren", "🧮 Exakt (Raster)"])
        with tab_v:
            st.markdown(T.METHODEN["drift"])
        with tab_e:
            st.markdown("Die geschlossene Form wird gegen die numerische Wasserstand-Lösung auf einem Raster (Zellbreite σ/24) verglichen, für eine Wolke bei pünktlichem Start.")
            if st.button("Raster nachrechnen", key="exakt_raster"):
                st.session_state["exakt_d"] = (s0, dif, su, t0, L0)
            if st.session_state.get("exakt_d") == (s0, dif, su, t0, L0):
                S0 = D.varianz(t0, *par)
                geschlossen = D.p_gauss(D.aufwand_fuer_L0(L0, S0), 1.0, S0)
                raster = D.p_raster(D.aufwand_fuer_L0(L0, S0), 1.0, S0, 0.0)
                st.metric("Raster minus geschlossene Form", f"{z(100 * (raster - geschlossen), 4)} Punkte", delta=f"geschlossen {z(100 * geschlossen, 3)} %, Raster {z(100 * raster, 3)} %", delta_color="off")
    _grenzen_und_mathe("drift")


# ================================================================================================ gemeinsame Expander
def _grenzen_und_mathe(ansicht: str):
    with st.expander("🚧 Wo die Annahmen enden"):
        st.markdown(T.GRENZEN[ansicht])
    with st.expander("📐 Mathematische Formulierung"):
        # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/suchaufwand-erklaert.json, nicht von Hand ändern)
        st.markdown(
            r"""
**Lesehilfe.** Gegeben sind eine Karte $p$, eine Entdeckungsstärke $\alpha$ und ein Budget (Ansichten Zuteilung, Nichtfund und Weg) beziehungsweise die Streuung einer Driftwolke (Ansicht Drift). Gesucht sind die Zuteilung der Durchgänge, die Karte nach einem Nichtfund, der Weg des Teams oder der Verlust bei späterem Start. Die Formeln stehen je Ansicht und gelten nur dort.

| Symbol | Bedeutung |
|---|---|
| $i = 1, \dots, N$ | Zelle des Gitters, $N$ ist die Zahl der Zellen |
| $p_i$ | Aufenthaltswahrscheinlichkeit in Zelle $i$ vor der Suche (Prior), Summe 1 |
| $\alpha$ | Entdeckungsstärke je Durchgang; ein Durchgang entdeckt ein Ziel in der Zelle mit Wahrscheinlichkeit $1 - e^{-\alpha}$ |
| $c_i, B$ | Aufwand eines Durchgangs in Zelle $i$ (ohne Gelände 1), Budget in Aufwandseinheiten |
| $z_i$ | Zahl der Durchgänge in Zelle $i$ (ganzzahlig, nicht negativ) |
| $P, \Delta_i$ | Entdeckungswahrscheinlichkeit, Zuwachs durch den nächsten Durchgang in Zelle $i$ (Grenzzuwachs) |
| $q_i$ | Wahrscheinlichkeit der Zelle $i$ nach einem Nichtfund (Posterior) |
| $P_1, P_2$ | Entdeckungswahrscheinlichkeit der Runde 1 und der Runde 2 (Ansicht Nichtfund) |
| $x_{t,i}, T, b$ | 1, wenn das Team im Schritt $t$ in Zelle $i$ sucht; Zahl der Zeitschritte; Basiszelle |
| $N(j)$ | Zelle $j$ und ihre Nachbarzellen (nicht die Zellenzahl $N$) |
| $S(t), s_0, D, s_U$ | Varianz der Wolke je Achse zur Zeit $t$; Streuung des Starts, Diffusion, Unsicherheit der konstanten Geschwindigkeit |
| $E, L$ | Suchaufwand (teilbar, Flächeneinheiten); $L = \sqrt{\alpha E / (\pi S)}$ |
| $t_0, d$ | Startzeit und relative Verzögerung; Verlust und Elastizität siehe letzte Formel |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/suchaufwand-erklaert.html#fachleute)
            """
        )
        # erklaerseite:lesehilfe:end
        st.markdown(T.MATHE_LESEHILFE[ansicht])
        for gl in T.MATHE_GLEICHUNGEN[ansicht]:
            st.latex(gl)
        st.markdown(T.MATHE_SATZ[ansicht])


ANSICHTEN = {"zuteilung": ansicht_zuteilung, "nichtfund": ansicht_nichtfund, "weg": ansicht_weg, "drift": ansicht_drift}
