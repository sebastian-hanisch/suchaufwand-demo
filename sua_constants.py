"""Konstanten: Ansichten, Reglerwerte, Farben und Presets (alle Instanzen und Zahlen erfunden)."""

SEED_MAX = 999999

# Die vier Ansichten (Umschalter): nur die gewählte Ansicht rechnet.
ANSICHTEN = ("zuteilung", "nichtfund", "weg", "drift")
ANSICHT_NAMEN = {
    "zuteilung": "1 · Zuteilung und Gelände",
    "nichtfund": "2 · Nichtfund und falsches α",
    "weg": "3 · Weg",
    "drift": "4 · Drift und Verzögerung",
}

PRIOR_NAMEN = {"ein_gipfel": "Ein Gipfel", "zwei_gipfel": "Zwei Gipfel (Driftwolke)", "flach": "Flach"}
PRIOR_SATZ = {"ein_gipfel": "ein Gipfel", "zwei_gipfel": "zwei Gipfel", "flach": "flacher Prior"}
PRIORS_ZWEI = ("ein_gipfel", "zwei_gipfel")
PRIORS_DREI = ("ein_gipfel", "zwei_gipfel", "flach")

# Ansicht 1
GITTER = (8, 12)
ALPHAS = (0.2, 0.5, 1.0, 2.0, 3.0, 10.0)
KAPPAS = (0.125, 0.25, 0.5, 1.0, 2.0)
TAUS = (0.5, 0.8, 0.95, 0.99)
GAMMAS = (0.0, 0.5, 1.0)
ALPHAS_GELAENDE = (0.5, 1.0, 3.0, 10.0)
GITTER_GELAENDE = 8
# Ansicht 2
ALPHAS_NICHTFUND = (0.5, 1.0, 2.0)
KAPPAS_NICHTFUND = (0.25, 0.5, 1.0, 2.0)
ANTEILE_RUNDE1 = (0.25, 0.5, 0.75)
VERFAHREN_RUNDE2 = ("aktualisiert", "alte_karte", "dort_weiter")
VERFAHREN_RUNDE2_NAMEN = {"aktualisiert": "auf der aktualisierten Karte", "alte_karte": "auf der alten Karte", "dort_weiter": "„dort weiter bis voll“"}
ALPHAS_ANGENOMMEN = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
ALPHA_WAHR = 1.0
GITTER_NICHTFUND = 8
# Ansicht 3
GITTER_WEG = 5
ALPHAS_WEG = (0.3, 1.0, 3.0, 10.0)
TS_WEG = (6, 10, 15)
VERFAHREN_WEG = ("exakt", "orient", "greedy_kurz", "greedy_ziel", "maehen_ganz", "maehen_rechteck")
VERFAHREN_WEG_NAMEN = {
    "exakt": "Wegoptimum (exakt, vorgerechnet)",
    "orient": "Orienteering-Fassung (vorgerechnet)",
    "greedy_kurz": "Greedy kurzsichtig",
    "greedy_ziel": "Greedy mit Ziel (neu planen)",
    "maehen_ganz": "Mähen über das ganze Gebiet",
    "maehen_rechteck": "Mähen im besten Rechteck",
}
INSTANZEN_WEG = tuple(range(1, 31))
ORIENTEERING_LINK = "https://github.com/sebastian-hanisch/gewinn-touren-demo"
# Ansicht 4
HYPOTHESEN = (1, 2)
VERZOEGERUNGEN = (0.10, 0.25, 0.50, 1.00)
L0_WERTE = (0.5, 1.0, 2.0, 4.0)
TEILCHEN_ANZEIGE = 2000

COLOR_KARTE = "Blues"
COLOR_ZUTEILUNG = "#2F6B65"
COLOR_WEG = "#c0392b"
COLOR_GRAU = "rgba(150,150,150,0.55)"
COLOR_LINIE = "#4c78a8"
COLORS_VERFAHREN = {
    "opt": "#2F6B65", "voll": "#e08a1e", "durchgang": "#8e5ea2", "rechteck": "#4c78a8",
    "aktualisiert": "#2F6B65", "alte_karte": "#8e5ea2", "dort_weiter": "#e08a1e",
}

HINWEIS = (
    "**Anwendungsbeispiel, keine Beratung.** Alle Instanzen, Karten, Zahlen und Zeitmaße sind erfunden und stammen aus einem Modell mit den Annahmen, die "
    "unter „Wo die Annahmen enden“ stehen. Die Demo sagt nichts über echte Einsätze und gibt keine Einsatzempfehlung."
)

PRESETS = {
    "Grundfall: ein Gipfel, α = 1, Budget = N": {"ansicht": "zuteilung", "z_prior": "ein_gipfel", "z_gitter": 8, "z_alpha": 1.0, "z_kappa": 1.0},
    "Zwei Gipfel, knappes Budget": {"ansicht": "zuteilung", "z_prior": "zwei_gipfel", "z_gitter": 8, "z_alpha": 1.0, "z_kappa": 0.125},
    "Sichere Entdeckung (α = 10, Gegenfall)": {"ansicht": "zuteilung", "z_prior": "ein_gipfel", "z_gitter": 8, "z_alpha": 10.0, "z_kappa": 1.0},
    "Flacher Prior (Gegenfall)": {"ansicht": "zuteilung", "z_prior": "flach", "z_gitter": 8, "z_alpha": 1.0, "z_kappa": 1.0},
    "Gelände γ = 1": {"ansicht": "zuteilung", "g_prior": "ein_gipfel", "g_gamma": 1.0, "g_alpha": 1.0, "g_kappa": 0.5},
    "Nichtfund: zwei Runden": {"ansicht": "nichtfund", "n_prior": "ein_gipfel", "n_alpha": 1.0, "n_kappa": 1.0, "n_anteil": 0.5, "n_verfahren": "aktualisiert"},
    "Falsches α: zu hoch": {"ansicht": "nichtfund", "f_prior": "ein_gipfel", "f_alpha": 10.0, "f_kappa": 1.0},
    "Weg: ein Gipfel, α = 1, T = 15": {"ansicht": "weg", "w_prior": "ein_gipfel", "w_alpha": 1.0, "w_T": 15, "w_verfahren": "exakt"},
    "Weg: zwei Gipfel": {"ansicht": "weg", "w_prior": "zwei_gipfel", "w_alpha": 1.0, "w_T": 15, "w_verfahren": "exakt"},
    "Weg: α = 10 (Orienteering)": {"ansicht": "weg", "w_prior": "ein_gipfel", "w_alpha": 10.0, "w_T": 10, "w_verfahren": "orient"},
    "See: gemischte Drift": {"ansicht": "drift", "d_s0": 1.0, "d_D": 0.5, "d_sU": 0.3},
    "See: nur unsicherer Strom": {"ansicht": "drift", "d_s0": 0.0, "d_D": 0.0, "d_sU": 0.3},
}
PRESET_HELP = {
    "Grundfall: ein Gipfel, α = 1, Budget = N": "8×8-Gitter, ein Gipfel, Entdeckungsstärke 1, Budget so viele Durchgänge wie Zellen: das Optimum gegen drei Abkürzungen.",
    "Zwei Gipfel, knappes Budget": "Prior aus einer Driftwolke mit zwei Hypothesen, nur ein Achtel so viele Durchgänge wie Zellen.",
    "Sichere Entdeckung (α = 10, Gegenfall)": "Ein Durchgang entdeckt fast sicher: in der Messreihe liegen alle Verfahren höchstens 0,01 Punkte auseinander.",
    "Flacher Prior (Gegenfall)": "Alle Zellen gleich wahrscheinlich: „ein Durchgang je Zelle“ ist hier das Optimum.",
    "Gelände γ = 1": "Aufwand je Durchgang schwankt im Gelände; Ordnen nach Prior je Aufwand gegen Ordnen nach Prior allein.",
    "Nichtfund: zwei Runden": "Zwei Runden mit je der Hälfte des Budgets; nach Runde 1 ohne Fund wird die Karte aktualisiert.",
    "Falsches α: zu hoch": "Zuteilung mit angenommener Entdeckungsstärke 10, bewertet mit der wahren Stärke 1.",
    "Weg: ein Gipfel, α = 1, T = 15": "5×5-Gitter, Team ab der Ecke, 15 Schritte; Wegoptimum gegen Greedy, Mähen und Orienteering-Fassung.",
    "Weg: zwei Gipfel": "Wie zuvor mit zwei Gipfeln: ein Rechteck deckt beide Wolken nicht ab.",
    "Weg: α = 10 (Orienteering)": "Sichere Entdeckung: jede Zelle zählt nur beim ersten Besuch, das ist ein Orienteering-Problem.",
    "See: gemischte Drift": "Start- und Strömungsunsicherheit mit Diffusion (s0 = 1, D = 0,5, sU = 0,3): Fläche und Verzögerungsverlust.",
    "See: nur unsicherer Strom": "Nur die je Teilchen konstante Geschwindigkeitsunsicherheit (D = 0): die Fläche wächst mit dem Quadrat der Zeit.",
}
