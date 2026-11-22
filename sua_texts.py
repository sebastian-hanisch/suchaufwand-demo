"""Feste Texte der App: Einleitung, Kernfragen, Erklärtexte der Verfahren, Grenzen und mathematische Formulierung.

Eine Fassung davon steht in der README (Modell, Methodik, Grenzen); `tests/test_texts.py` hält beide Fassungen zusammen. Alle Instanzen erfunden.
"""

import sua_constants as C

TITEL = "Suchaufwand: wo sucht man, wenn nichts gefunden wurde"

EINLEITUNG = f"""
{C.HINWEIS}

Ein Team sucht in Gelände oder auf See ein **ruhendes Ziel**, dessen Ort nur als **Wahrscheinlichkeitskarte** (englisch Probability Map, Prior) bekannt ist. Die Demo fragt in vier Ansichten,
**wohin der nächste Durchgang gehört**, was ein **Nichtfund** an der Karte ändert, was der **Weg** des Teams kostet und wie schnell eine **Driftwolke** wächst
(Kulisse Berg- und Seerettung, alles erfunden). Weiter unten: die **Kernfrage** der Ansicht und die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**,
**🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""

KERNFRAGEN = {
    "zuteilung": "Wo steckt man den nächsten Durchgang hin, und wie viel kostet die Abkürzung?",
    "nichtfund": "Was ändert ein Nichtfund an der Karte, und wann schadet ein falsches α?",
    "weg": "Was kostet der Weg, und hilft „einfach mähen“?",
    "drift": "Wie schnell wächst die Wolke, und was kostet späteres Starten?",
}

ANSICHT_EINLEITUNG = {
    "zuteilung": (
        "**Modell:** Das Ziel ruht in einer von N Zellen eines Gitters, die Karte p gibt die Aufenthaltswahrscheinlichkeit je Zelle. Ein **Durchgang** in einer Zelle kostet eine Aufwandseinheit "
        "(im Gelände mehr) und entdeckt das Ziel, falls es dort ist, mit der Wahrscheinlichkeit 1 − e^(−α); α ist die **Entdeckungsstärke je Durchgang**. "
        "Der Aufwand wird in **ganzen Durchgängen** gezählt. **Lücke** = Optimum minus Verfahren in **Prozentpunkten Entdeckungswahrscheinlichkeit** (nie Prozent des Optimums)."
    ),
    "nichtfund": (
        "**Modell wie in Ansicht 1.** Nach einem **Nichtfund** ändert sich die Karte: Zellen, in denen schon gesucht wurde, werden unwahrscheinlicher (Bayes-Regel). "
        "Die Ansicht rechnet zwei Runden mit je der Hälfte des Budgets und fragt, was man in Runde 2 mit dieser Karte anfängt, und was ein falsch angenommenes α kostet."
    ),
    "weg": (
        "**Modell wie in Ansicht 1, aber das Team muss laufen:** auf einem 5×5-Gitter ab der Basis (oben links) steht es in jedem Zeitschritt in einer Zelle, sucht dort einen Durchgang "
        "und geht danach zu einer Nachbarzelle oder bleibt. Die **Anfahrt zählt als Suchen**. Wegoptimum und Orienteering-Fassung sind mit CP-SAT **vorgerechnet** (eingefroren)."
    ),
    "drift": (
        "**Modell (einheitenlos, erfunden):** Ein Teilchen startet bei einer unsicheren Position, treibt mit einer je Teilchen unsicheren, aber konstanten Geschwindigkeit und diffundiert zusätzlich. "
        "Die Wolke bleibt eine Gauß-Verteilung mit der Varianz S(t) je Achse; die Fläche eines Bereichs mit festem Wahrscheinlichkeitsanteil ist proportional zu S(t)."
    ),
}

# ----------------------------------------------------------------------------------------- Methodenvergleich (je Ansicht)
METHODEN = {
    "zuteilung": """
**Optimum (Grenzzuwachs):** Jeder nächste Durchgang geht in die Zelle mit dem größten Zuwachs der Entdeckungswahrscheinlichkeit, p_i · e^(−α z_i) · (1 − e^(−α)).
Bei gleichem Preis je Zelle ist das dasselbe wie die Zelle mit dem größten aktuellen Posterior. Durch Aufzählung aller Zuteilungen ist die Zuteilung in diesem Modell optimal
(Zahl der Fälle in der README); für stetigen Aufwand ist die Zuteilung auf gleichen Grenzwert die Optimalitätsbedingung (Lagrange, Kuhn-Tucker).

**„Ein Durchgang je Zelle“:** reihum in der Reihenfolge der Prior-Werte, bis das Budget aufgebraucht ist (bei Budget unter N die wahrscheinlichsten Zellen). Das ist die gleichmäßige Zuteilung.

**„Vollständig“:** Zellen in der Reihenfolge der Prior-Werte je so oft absuchen, dass die Entdeckung in der Zelle mindestens τ erreicht, dann die nächste. Das ist **eine Familie** (τ von 0,5 bis 0,99),
kein einzelnes Verfahren; je nach Budget und α gewinnt ein anderes τ (bei 0,25·N meist 0,5, bei N meist 0,95), das Optimum braucht keins.

**Bestes Rechteck, gleichmäßig:** unter 6 vorausgewählten Rechtecken des Gitters (Vorauswahl nach dem Wert bei gleichmäßiger Verteilung des Budgets; Art des Planers in SAROPS) das mit dem größten wahren Wert, Rest auf die wahrscheinlichsten Zellen. In Ansicht 3 ist „bestes Rechteck“ dagegen das nach dem wahren Wert beste aller Rechtecke.

**Obergrenze:** bei **teilbarem** Aufwand (Wasserstand-Lösung) ist das Optimum größer als bei ganzen Durchgängen; die gestrichelte Marke ist nur diese Obergrenze.

**Gelände:** Durchgänge kosten je Zelle verschieden viel (ganzzahlig, im Mittel etwa 2). „Prior je Aufwand“ ordnet die Zellen nach p_i / c_i. Der **aktualisierende Greedy** nimmt stets den größten Zuwachs je Kosten;
bei ungleichen Kosten ist das nur eine Näherung (Rucksack-Problem), das **exakte Optimum** kommt aus einer Rucksack-Rechnung über Zellen und Durchgänge.
""",
    "nichtfund": """
**Posterior nach Nichtfund:** q_i = p_i · e^(−α z_i) / (1 − P). Zellen mit vielen Durchgängen werden unwahrscheinlicher, unbesuchte relativ wahrscheinlicher.

**Runde 2 auf der aktualisierten Karte** ist der Grenzzuwachs-Greedy auf dem Posterior; zusammen mit Runde 1 gibt das eine **Gesamtzuteilung mit demselben Wert** wie ein Vorgehen mit dem ganzen Budget auf dem Prior (additives Prinzip).
**Alte Karte:** die Zuteilung der Runde 1 wird mit dem Budget der Runde 2 auf dem Prior wiederholt. **„Dort weiter bis voll“:** die Zellen mit dem größten Posterior werden je bis 95 % abgesucht.
**Falsches α:** die Zuteilung wird mit einer angenommenen Entdeckungsstärke berechnet und mit der wahren bewertet.

**Schwelle auf dem Grenzzuwachs:** Nach der optimalen ganzzahligen Zuteilung ist der kleinste Grenzzuwachs eines genommenen Durchgangs nicht kleiner als der größte Grenzzuwachs des nächsten (exakt).
Der Posterior der besuchten Zellen liegt nur in einem Band vom Faktor e^α, und unbesuchte Zellen können über besuchten liegen; „alle abgesuchten Zellen gleich hoch“ gilt nur für stetigen Aufwand.
""",
    "weg": """
**Greedy kurzsichtig:** in jedem Schritt unter Bleiben und den Nachbarzellen die mit dem größten Grenzzuwachs. **Greedy mit Ziel:** für jedes erreichbare Ziel die Summe der Grenzzuwächse auf dem besten kürzesten Weg durch die Schrittzahl;
das beste Verhältnis gewinnt, es wird nach jedem Schritt neu bestimmt. **Mähen über das ganze Gebiet:** Schlangenlinie ab der Ecke, danach hin und her. **Mähen im besten Rechteck:** das beste Rechteck, in Schlangenlinie gemäht, mit Anfahrt.

**Wegoptimum:** ganzzahliges Modell, mit CP-SAT vorgerechnet; Instanzen ohne Beweis sind markiert („beste gefundene Lösung“). **Wegpreis** = Optimum ohne Weg (die T Durchgänge frei auf Zellen verteilbar) minus Wegoptimum.
**Orienteering-Fassung:** jede Zelle zählt nur beim ersten Besuch (Prämie p_i, kein Rückweg); unter allen ihren Optima wird der Weg mit dem höchsten wahren Wert genommen (**günstigster Fall**). Die **Überschätzung** ist die Prämien-Summe des Weges
minus seine wahre Entdeckungswahrscheinlichkeit.
""",
    "drift": """
**Geschlossene Form** für eine Gauß-Wolke bei optimal verteiltem, teilbarem Aufwand E: P = 1 − (1 + L) · e^(−L) mit L = √(α E / (π S)). **Verzögerung:** E bleibt gleich, die Wolke ist inzwischen größer.
**Verlust** = 1 − P(verzögert) / P(pünktlich). **Elastizität** = −ln(P₁/P₀) / ln(t₁/t₀): unter 1 fällt P prozentual langsamer, als die Startzeit steigt.
**Zwei Hypothesen:** zwei Wolken gleichen Gewichts im Abstand 2·dv·t (Querwind links und rechts); die optimale Verteilung wird numerisch auf einem Raster gelöst (Wasserstand-Lösung).
""",
}

# ----------------------------------------------------------------------------------------- Wo die Annahmen enden
GRENZEN_ALLGEMEIN = """
- **Alles erfunden.** Karten, Instanzen, Zeitmaße und Parameter sind erfunden; keine Feldversuchsdaten, keine Einsatzempfehlung.
- **Entdeckungsfunktion** 1 − e^(−α z): Zufallssuche mit ganzen Durchgängen und einer Stärke α je Durchgang. Andere Sensorformen (feste Reichweite, inverse Würfelfunktion) sind **nicht gemessen**; die Aussagen gelten nicht für jede Sensorform.
- **Ruhendes Ziel:** der Ort des Ziels ändert sich während der Suche nicht; bewegte Ziele, zerreißende Verteilungen, mehrere Teams, Hubschrauber und Sensoren mit Reichweite sind nicht Teil der Rechnung.
- **Zähleinheit** der Lücke: Prozentpunkte Entdeckungswahrscheinlichkeit; Mittel über 30 erfundene Instanzen je Zelle mit 95-%-Bootstrap-Intervall der Mittelwerte.
"""

GRENZEN = {
    "zuteilung": GRENZEN_ALLGEMEIN + """
- **Optimum ganzzahlig** gilt bei gleichem Preis je Zelle; für ungleiche Kosten ist der Greedy nur eine Näherung (Rucksack-Lücke, siehe Verdikt).
- **Gelände und Prior sind unabhängig erzeugt.** Liegt der Gipfel im schwierigen Gelände, ist das **nicht gemessen**.
- **Gitter:** 8×8 und 12×12; im Gelände nur 8×8.
""",
    "nichtfund": GRENZEN_ALLGEMEIN + """
- **„Aktualisieren bringt 0“** gilt im Wert (derselbe Gesamtwert) nur bei **frei verteilbarem Aufwand** und gleichem Preis je Zelle: jede Runde darf jede Zelle beliebig bedienen. Im Rechteck-Muster ist das kurzsichtig aktualisierte Vorgehen nicht mehr optimal.
- **Falsches α** ist nur für den wahren Wert α = 1 und acht Zellen gemessen; die Wirkung eines Nichtfunds hängt von der Objektart ab (nach SAROPS: Floß gegen Person im Wasser).
- Die Karte nach Nichtfund bleibt hier eine Karte; ein Ziel, das **seinen Zustand ändert** (Boot, Rettungsinsel, Person im Wasser), braucht ein Mehrzustandsmodell und ändert, welche Gebiete zuerst abgesucht werden. Die Entdeckungswahrscheinlichkeit der Demo wertet alle Funde gleich.
""",
    "weg": GRENZEN_ALLGEMEIN + """
- **Gitter 5×5, Basis in der Ecke, kein Rückweg**; 8×8 nur als vorgerechnete Tabelle. **Anfahrt zählt als Suchen**: in jedem Schritt wird gesucht. Bei getrennten Bewegungs- und Suchschritten (Totzeit) gilt ein anderes Modell, in dem es einen optimalen Weg ohne Rückkehr zu einem Punkt gibt; das ist hier **nicht gemessen**.
- **Mehrere Durchgänge in derselben Zelle** („Bleiben oder Rückkehr“) sind das, was die Orienteering-Fassung nicht kennt; die Lücke der Fassung ist ein **Modellunterschied**, kein Vorwurf an ein Orienteering-Verfahren.
- **Zwei Gipfel:** vermutlich deckt ein Rechteck zwei getrennte Wolken nicht ab (nicht eigens getestet); „bestes Rechteck“ ist hier nach dem wahren Wert gewählt, der günstigste Fall für das Verfahren; in der Praxis setzt man bei mehreren Sucheinheiten mehrere Rechtecke, das ist **nicht gemessen**.
- **Härte:** Die Maximierung der Entdeckungswahrscheinlichkeit auf Wegen ist in allgemeinen Graphen NP-schwer; für das Gitter wird nichts behauptet. Unbewiesene Lösungen sind markiert, die Lücken der Verfahren sind dann eher zu klein.
- **Zählt ein früher Fund mehr als ein später** (Lawine, Seenot), ändert das in dieser Rechnung (ruhendes Ziel, dessen Zustand sich nicht ändert, ohne Weg) die Reihenfolge der Durchgänge, nicht die abgesuchten Zellen. Ändert sich der Zustand des Ziels, ändert sich auch, welche Gebiete zuerst abgesucht werden, bis zu einem Verlust an Gesamterfolg.
  Bei schnell fallender Überlebensfähigkeit ist die Minimierung der Zeit bis zum Fund das angemessenere Kriterium. Eine Zahl dazu steht hier nicht.
""",
    "drift": GRENZEN_ALLGEMEIN + """
- **Gauß-Wolke, konstantes Strömungsfeld, isotrope Streuung.** Zerreißt die Verteilung (etwa an einer Strömungsgrenze), gelten die Formeln nicht.
- **Feste Seite der Querwind-Hypothesen:** der Zwei-Wolken-Fall nimmt an, dass kein Teilchen die Seite wechselt. Wechsel („Halsen“) füllen in Wirklichkeit die Lücke zwischen den Wolken; mit Seitenwechsel ist das **nicht gemessen**.
- **Teilbarer Aufwand** und eine feste Suchleistung E; die Verluste sind relativ zu P bei pünktlichem Start.
- **Einheitenlos:** Zeit, Länge und Parameter haben keine Einheit und stammen nicht aus Feldversuchen. „Nur unsicherer Strom“ heißt: nur die je Teilchen konstante Geschwindigkeitsunsicherheit sU.
- Das Modell nennt die Fläche ∝ Zeit nur für reine Diffusion (s₀ = 0, sU = 0) richtig; mit überwiegender konstanter Geschwindigkeitsunsicherheit wächst sie schneller, bis zu t². Das ist ein Schluss aus dem Modell, keine Aussage über die Wirklichkeit.
""",
}

# ----------------------------------------------------------------------------------------- Mathematische Formulierung
MATHE_LESEHILFE = {
    "zuteilung": """
**Gegeben:** Karte p (Summe 1), Entdeckungsstärke α, Kosten c_i je Durchgang, Budget B. **Gesucht:** ganzzahlige Zahl z_i von Durchgängen je Zelle mit größter Entdeckungswahrscheinlichkeit.

| Symbol | Bedeutung |
|---|---|
| i = 1…N | Zelle des Gitters, N ist die Zahl der Zellen |
| p_i | Aufenthaltswahrscheinlichkeit (Prior), Summe 1 |
| α | Entdeckungsstärke je Durchgang |
| c_i | Aufwand eines Durchgangs in Zelle i (ohne Gelände 1) |
| B | Budget in Aufwandseinheiten |
| z_i | Zahl der Durchgänge in Zelle i (ganzzahlig, ≥ 0) |
| P | Entdeckungswahrscheinlichkeit |
""",
    "nichtfund": """
**Gegeben:** Prior p, Zuteilung z der Runde 1 und Nichtfund. **Gesucht:** Posterior und Zuteilung der Runde 2.

| Symbol | Bedeutung |
|---|---|
| p_i, z_i, α | wie in Ansicht 1 |
| P, P₁, P₂ | Entdeckungswahrscheinlichkeit der Runde 1 (P = P₁); P₂ die der Runde 2 auf der Karte q |
| q_i | Posterior nach Nichtfund |
| α_ang | angenommene Entdeckungsstärke (falsches α) |
""",
    "weg": """
**Gegeben:** Karte p, α, Zahl T der Zeitschritte, Basiszelle b, Nachbarschaft N(j) (Zelle j selbst und ihre vier Nachbarn). **Gesucht:** der Weg mit der größten Entdeckungswahrscheinlichkeit.

| Symbol | Bedeutung |
|---|---|
| x_(t,i) | 1, wenn das Team im Schritt t in Zelle i sucht |
| z_i | Zahl der Schritte in Zelle i |
| N(j) | Zelle j und ihre Nachbarzellen (nicht die Zellenzahl N) |
| b | Basiszelle (Zelle 0) |
""",
    "drift": """
**Gegeben:** Streuung s₀ des Starts, Diffusion D, Geschwindigkeitsunsicherheit sU, Aufwand E, Startzeit t₀ und Verzögerung d. **Gesucht:** Fläche, Entdeckungswahrscheinlichkeit und Verlust.

| Symbol | Bedeutung |
|---|---|
| S(t) | Varianz der Wolke je Achse zur Zeit t |
| s₀, D, sU | Streuung des Starts, Diffusion, Unsicherheit der konstanten Geschwindigkeit |
| E | Suchaufwand (teilbar, Flächeneinheiten) |
| L | √(α E / (π S)) |
| P | optimale Entdeckungswahrscheinlichkeit |
| t₀, d, P₀, P₁ | Startzeit, relative Verzögerung; P₀ und P₁ die Werte beim Start t₀ und beim späteren Start t₁ = t₀(1 + d) (nicht die Runden der Ansicht Nichtfund) |
""",
}

# je Gleichung ein eigener Block (KaTeX); der Satz darunter sagt, ob das Modell vollständig ist oder was fehlt
MATHE_GLEICHUNGEN = {
    "zuteilung": [
        r"\max_{z}\; P(z)=\sum_{i=1}^{N} p_i\bigl(1-e^{-\alpha z_i}\bigr)",
        r"\text{u. d. N.}\quad \sum_{i=1}^{N} c_i z_i \le B,\qquad z_i\in\mathbb{N}_0",
        r"\Delta_i(z)=p_i\,e^{-\alpha z_i}\bigl(1-e^{-\alpha}\bigr)\quad\text{(Grenzzuwachs des nächsten Durchgangs)}",
    ],
    "nichtfund": [
        r"q_i=\frac{p_i\,e^{-\alpha z_i}}{1-P},\qquad P=\sum_i p_i\bigl(1-e^{-\alpha z_i}\bigr)",
        r"\Delta_i=\bigl(1-e^{-\alpha}\bigr)\,(1-P)\,q_i\quad\text{(ganze Durchgänge, gleicher Preis)}",
        r"P_{\text{gesamt}}=P_1+(1-P_1)\,P_2(q),\qquad P_{\text{gesamt}}=P(z_1+z_2)",
    ],
    "weg": [
        r"\max\; \sum_{i} p_i\bigl(1-e^{-\alpha z_i}\bigr),\qquad z_i=\sum_{t=0}^{T-1}x_{t,i}",
        r"\sum_i x_{t,i}=1\;(t=0,\dots,T-1),\qquad x_{0,b}=1,\qquad x_{t+1,j}\le\sum_{i\in N(j)}x_{t,i},\qquad x_{t,i}\in\{0,1\}",
    ],
    "drift": [
        r"S(t)=s_0^2+2Dt+s_U^2t^2",
        r"L=\sqrt{\frac{\alpha E}{\pi S}},\qquad P=1-(1+L)\,e^{-L}",
        r"\text{Verlust}(d)=1-\frac{P(t_0(1+d))}{P(t_0)},\qquad \text{Elastizität}=-\frac{\ln\bigl(P_1/P_0\bigr)}{\ln(t_1/t_0)}",
    ],
}
MATHE_SATZ = {
    "zuteilung": "Das ist das vollständige Modell der Ansicht (ein ganzzahliges Rucksack-Problem mit konkaver Zielfunktion je Zelle). Bei gleichem Preis löst der Greedy es exakt, bei ungleichen Kosten rechnet die Demo es mit einer Rucksack-Rechnung.",
    "nichtfund": "Vollständig für zwei Runden mit gleichem Preis je Zelle; das Rechteck-Muster ist ein eingeschränkter Fall, den diese Formeln nicht abbilden (die Demo rechnet ihn durch Aufzählung der Rechtecke).",
    "weg": "Vollständig für den Weg auf dem Gitter (CP-SAT rechnet den Zielterm über Hilfsvariablen je Zelle und Besuch, deren Gewichte mit jedem Besuch fallen). Die Orienteering-Fassung ersetzt das Ziel durch die Prior-Masse der besuchten Zellen.",
    "drift": "**Skizze der Herleitung:** Die Wolke ist normalverteilt mit der Varianz S(t) je Achse; das Optimum bei teilbarem Aufwand ist die Wasserstand-Lösung (alle besuchten Stellen auf gleichem Grenzwert), über Radien integriert ergibt sich die geschlossene Form. Die Zwei-Hypothesen-Rechnung löst dieselbe Wasserstand-Aufgabe numerisch auf einem Raster.",
}
