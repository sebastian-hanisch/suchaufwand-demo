# Suchaufwand: wo sucht man, wenn nichts gefunden wurde – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-suchaufwand-demo.streamlit.app/)**

**Anwendungsbeispiel, keine Beratung.** Alle Instanzen, Karten, Zahlen und Zeitmaße sind erfunden und stammen aus einem Modell mit den Annahmen, die unter „Ehrliche Grenzen“ stehen. Die Demo sagt nichts über echte Einsätze und gibt keine Einsatzempfehlung.

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`). Ein Team sucht in Gelände oder auf See ein **ruhendes Ziel**, dessen Ort nur als Wahrscheinlichkeitskarte (englisch Probability Map) bekannt ist (Kulisse Berg- und Seerettung). Vier Ansichten, per Umschalter gewählt (nur die gewählte rechnet): **1 Zuteilung und Gelände**, **2 Nichtfund und falsches α**, **3 Weg**, **4 Drift und Verzögerung**.

## Warum dieses Problem

Nach einem Nichtfund bleibt die Frage, wo der nächste Durchgang hingehört. Die Demo beantwortet vier Teilfragen: Wo steckt man den nächsten Durchgang hin, und wie viel kostet die Abkürzung? Was ändert ein Nichtfund an der Karte, und wann schadet ein falsches α? Was kostet der Weg des Teams, und hilft „einfach mähen“? Wie schnell wächst eine Driftwolke, und was kostet späteres Starten?

## Modell

Ruhendes Ziel in einer von N Zellen eines n×n-Gitters; **Prior** p_i (Summe 1) = Aufenthaltswahrscheinlichkeit vor der Suche. Aufwand wird in **ganzen Durchgängen** gezählt: ein Durchgang in Zelle i kostet eine Einheit (im Gelände c_i Einheiten) und entdeckt das Ziel, falls es dort ist, mit der Wahrscheinlichkeit 1 − e^(−α); α ist die **Entdeckungsstärke je Durchgang**. Bei z_i Durchgängen entdeckt die Suche mit 1 − e^(−α·z_i) (Zufallssuche). Die **Entdeckungswahrscheinlichkeit** (englisch Probability of Detection) ist P = Σ p_i (1 − e^(−α·z_i)), das Budget B zählt ohne Gelände Durchgänge (auch als κ = B/N); im Gelände ist es das κ-fache der Gesamtkosten. Der **Grenzzuwachs** des nächsten Durchgangs in Zelle i ist p_i·e^(−α·z_i)·(1 − e^(−α)); der **Posterior nach Nichtfund** ist q_i = p_i·e^(−α·z_i)/(1 − P). Für ganze Durchgänge gilt Grenzzuwachs = (1 − e^(−α))·(1 − P)·Posterior_i, für teilbaren Aufwand α·(1 − P)·Posterior_i. Bei gleichem Preis je Zelle ist „höchster Grenzzuwachs“ also dasselbe wie „höchster aktueller Posterior“.

**Lücke** = Optimum minus Verfahren in **Prozentpunkten Entdeckungswahrscheinlichkeit** (nie Prozent des Optimums), Mittel über 30 erfundene Instanzen je Zelle mit 95-%-Bootstrap-Intervall der Mittelwerte. Das **Optimum** ist bei gleichem Preis je Zelle die ganzzahlige Zuteilung des Grenzzuwachs-Greedy (durch Aufzählung belegt, keine gelesene Quelle). Der **Wegpreis** ist das Optimum ohne Weg (T Durchgänge frei auf Zellen verteilbar) minus das exakte Wegoptimum. Der Zufalls-Seed bestimmt die erfundene Instanz; Priors: „ein Gipfel“, „zwei Gipfel“ (aus Driftteilchen mit zwei Hypothesen gezählt) und „flach“. Die vollständige Formulierung steht im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

- **Optimum:** Grenzzuwachs-Greedy auf ganzen Durchgängen (Gleichstand: kleinster Index, Grenzzuwachs auf 12 Stellen gerundet). **Obergrenze:** die Wasserstand-Lösung bei teilbarem Aufwand (Bedingungen von Lagrange und Kuhn-Tucker, geschlossen); sie dient nur als Marke.
- **Abkürzungen (Ansicht 1):** „ein Durchgang je Zelle“ (reihum in der Reihenfolge der Prior-Werte), „vollständig“ (Zellen der Reihe nach je so lange absuchen, bis die Entdeckung in der Zelle mindestens τ erreicht, τ = 0,95; **eine Familie** mit τ von 0,5 bis 0,99) und das **beste Rechteck** bei gleichmäßiger Verteilung (Art des Planers in SAROPS; gemeint ist das beste von 6 vorausgewählten Rechtecken, die Vorauswahl folgt dem Wert bei gleichmäßiger Verteilung; in Ansicht 3 dagegen das nach dem wahren Wert beste aller Rechtecke).
- **Gelände:** ganzzahlige Kosten (Gelände aus einem geglätteten Zufallsfeld, Kosten im Mittel etwa 2, mindestens 1), Ordnen nach Prior oder nach Prior je Aufwand, aktualisierender Greedy nach Zuwachs je Kosten, exaktes Optimum aus einer Rucksack-Rechnung über Zellen und Durchgänge.
- **Nichtfund (Ansicht 2):** zwei Runden mit je der Hälfte des Budgets; Runde 2 auf der aktualisierten Karte, auf der alten Karte oder „dort weiter bis voll“; falsches α; im Rechteck-Muster (je Runde ein ganz abgedecktes Rechteck) kurzsichtig oder mit Vorausschau über die 40 besten Rechtecke der Runde 1.
- **Weg (Ansicht 3):** 5×5-Gitter ab der Ecke, in jedem Zeitschritt ein Durchgang an der aktuellen Zelle (die Anfahrt zählt als Suchen); Greedy kurzsichtig, Greedy mit Ziel, Mähen über das ganze Gebiet, Mähen im besten Rechteck, Wegoptimum und Orienteering-Fassung mit CP-SAT **vorgerechnet und eingefroren**.
- **Drift (Ansicht 4):** Gauß-Wolke mit S(t) = s₀² + 2Dt + sU²t², geschlossene Form P = 1 − (1 + L)e^(−L) mit L = √(αE/(πS)) bei teilbarem Aufwand, Zwei-Hypothesen-Wolke numerisch auf einem Raster; 30 erfundene Szenarien mit 40 000 Teilchen sind vorgerechnet.
- **Vorgerechnet** (Ordner `data/`, Werkzeuge `tools/sua_make_*.py` mit `--check`): die Messreihen der Ansichten 1 und 2 (`sua_messreihe.json`), die CP-SAT-Wege (`sua_weg.json`) und die Drift-Szenarien (`sua_drift.json`). Die App rechnet nur eine Instanz live.
- **Zufall:** nur der ganzzahlige SplitMix64 (`sua_rng.py`, `sua_zufall.py`). Ganzzahlen und Gleichverteilung sind plattformgleich; die Normalzahlen (Box-Muller) hängen im letzten Bit von der Mathematikbibliothek ab, deshalb sind alle Neu-Rechnen-Tests Bänder.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten **im erfundenen Modell, bei diesen Parametern und in der genannten Zähleinheit**. Wo „Mittel über 30 Instanzen“ steht, sind es die Zufalls-Seeds 1 bis 30; die Zahlen stammen aus den eingefrorenen Messreihen, den Formeln der Demo und ihren Tests (Testtoleranzen sind keine Messwerte); die Zahlen der Messreihen werden in `tests/test_claims.py` aus `data/` gelesen und, wo genannt, im Band neu gerechnet.

**Ansicht 1 – Zuteilung.** Mittlere Lücke in Prozentpunkten, Mittel über 20 Zellen (8×8 und 12×12, ein und zwei Gipfel, fünf Budgets von 0,125·N bis 2·N, je Zelle 30 Instanzen), in Klammern der größte Zellenmittelwert:

| Verfahren | α = 0,2 | 0,5 | 1,0 | 2,0 | 3,0 | 10 |
|---|---|---|---|---|---|---|
| vollständig (τ = 0,95) | 8,9 (14,5) | 8,3 (14,6) | 6,7 (13,6) | 6,5 (15,9) | 1,5 (4,9) | 0,0 |
| ein Durchgang je Zelle | 25,4 (51,9) | 23,1 (49,1) | 14,2 (33,1) | 4,6 (12,9) | 1,5 (4,9) | 0,0 |
| bestes Rechteck | 6,2 (20,0) | 6,6 (19,4) | 5,3 (17,5) | 2,4 (11,2) | 0,8 (4,3) | 0,0 |

- Das Intervall der Lücke liegt für α ≤ 3 in 95 von 100 Zellen („vollständig“), 93 von 100 („ein Durchgang“) und 95 von 100 (Rechteck) über 0. Bei α = 10 liegt die Lücke in allen 30 Zellen unter 0,01 Punkten; bei 2·N und α ≥ 3 sind alle Verfahren höchstens 0,25 Punkte hinter dem Optimum.
- **Kernzeile** (8×8, ein Gipfel, α = 1, Budget 0,125 / 0,25 / 0,5 / 1 / 2·N): Optimum 39,5 / 57,8 / 76,4 / 90,6 / 97,8 %; Lücke „ein Durchgang“ 0,5 / 4,2 / 14,7 / 27,4 [25,3; 29,4] / 11,3, „vollständig“ 11,6 / 10,7 / 7,3 / 3,0 / 3,3, Rechteck 0,2 / 0,6 / 1,3 / 1,3 / 0,8.
- **Nie abgesuchte Zellen** (8×8, ein Gipfel, α = 1, Mittel über 30 Instanzen): bei Budget N lässt das Optimum 57 % der Zellen unberührt, darin 2,5 % der Prior-Masse; bei Budget 0,125·N 88 % mit 40 %. „Ein Durchgang je Zelle“ lässt ab Budget N keine Zelle aus, das ist Absicht.
- **Bei zwei Gipfeln und Budget 0,125·N** (8×8, α = 1) liegt das beste Rechteck 9,7 Punkte hinter dem Optimum, „vollständig“ 9,0; bei einem Gipfel waren es 0,2 (Rechteck) und 11,6 („vollständig“): Das Rechteck wird schlechter, „vollständig“ besser.
- **„Vollständig“ ist eine Familie:** über τ ∈ {0,5; 0,8; 0,95; 0,99} (8×8, 72 Zellen) liegt die Lücke zwischen 0,6 und 31,9 Punkten; je nach Budget und α gewinnt ein anderes τ (bei 0,25·N meist 0,5, bei N meist 0,95, 0,99 in keiner der 18 Kombinationen aus Prior, α und Budget), das Optimum braucht keins.
- **Flacher Prior** (30 Zellen): „ein Durchgang je Zelle“ ist das Optimum (Lücke genau 0); „vollständig“ verliert bis 37,4 Punkte.
- **Obergrenze bei teilbarem Aufwand:** im Mittel über alle 180 Zellen (einschließlich flachem Prior) 5,1 Punkte über dem ganzzahligen Optimum, ohne flachen Prior (120 Zellen) 2,2 (je α = 0,2 / 0,5 / 1 / 2 / 3 / 10 mit flachem Prior: 0,10 / 0,53 / 1,65 / 4,41 / 7,15 / 16,62, ohne: 0,05 / 0,25 / 0,68 / 1,72 / 2,87 / 7,44); der größte Zellenwert 66,8 gehört zum flachen Prior bei α = 10 und Budget 0,25·N, ohne flachen Prior ist er 35,1. Der Mittelwert wird also vom flachen Prior und von α = 10 getrieben und ist nicht typisch.
- **Aufzählung:** in 120 Fällen (4×4 Budget 3, 5, 8: 90; 6×6 Budget 3, 4: 30) ist die Zuteilung des Grenzzuwachs-Greedy gleich dem besten Wert aller Zuteilungen, größte Abweichung 0.
- **Schwelle auf dem Grenzzuwachs:** nach der optimalen ganzzahligen Zuteilung ist der kleinste Grenzzuwachs eines genommenen Durchgangs nicht kleiner als der größte des nächsten (1080 von 1080 Instanzen, 8×8, 3 Priors × 4 α × 3 Budgets). Der Posterior der besuchten Zellen liegt nur im Band vom Faktor e^α (größter Quotient gleich e^α, nur bei Gleichstand der Prior-Werte erreicht, nie darüber); in 911 von 1080 Instanzen (671 von 720 ohne flachen Prior) liegt eine unbesuchte Zelle im Posterior **über** einer besuchten. Nur bei teilbarem Aufwand stehen alle besuchten Zellen im Posterior gleich hoch (größte relative Abweichung 4,5·10⁻¹⁵).

**Ansicht 1 – Gelände** (8×8, 40 Zellen je Geländefaktor = 2 Priors × 4 α × 5 Budgets, 30 Instanzen je Zelle; Kosten im Mittel etwa 2, Variationskoeffizient im Mittel 0,53 bei γ = 0,5 und 1,03 bei γ = 1,0):

- Ordnen nach **Prior je Aufwand** statt nach Prior allein gewinnt bei „vollständig“ im Mittel 1,1 Punkte (γ = 0,5) bzw. 2,0 (γ = 1,0), größter Zellenmittelwert 5,1; das Intervall liegt in 30 von 40 Zellen über 0, in keiner darunter. Bei „ein Durchgang“: 0,4 bzw. 0,7 Punkte, über 0 in 24 bzw. 19 von 40 Zellen, in keiner darunter. Bei γ = 0 sind beide Reihenfolgen identisch (Differenz 0 in 40 von 40 Zellen).
- Der **aktualisierende Greedy nach Zuwachs je Kosten** liegt im Mittel 0,02 (γ = 0,5) bzw. 0,05 (γ = 1,0) Punkte hinter dem exakten Optimum, größter Zellenmittelwert 0,29. Es bleibt eine **Rucksack-Lücke**: in 1 von 30 kleinen Instanzen (3×3, Kosten 1 bis 3, Budget 8) liegt der Greedy unter dem Optimum, größte Lücke 0,47 Punkte; das ist eine einzelne Stichprobe und keine Obergrenze (die frühere Stichprobe gab 5 von 30 Fällen und bis 4,96 Punkte).
- **Gegenfall:** bei α = 10 und γ = 1,0 bleibt der Vorteil des Verhältnisses bei knappem Budget bestehen (8×8, ein Gipfel, Budget 0,125 der Gesamtkosten: 2,8 Punkte), vermutlich weil dann das Auswahlproblem (Rucksack) übrig bleibt (die Ursache ist nicht eigens gemessen); bei Budget gleich den Gesamtkosten und darüber ist er verschwunden (Differenz 0,0). Gelände und Prior sind unabhängig erzeugt; liegt der Gipfel im schwierigen Gelände, ist das **nicht gemessen**.

**Ansicht 2 – Nichtfund** (8×8, zwei Runden, 24 Zellen = 2 Priors × 3 α × 4 Budgets):

- **Aktualisieren in Runden ist eine Zuteilung mit dem ganzen Budget:** ganzzahlig exakt gleich in 24 von 24 Zellen (im Wert, bei gleichem Preis je Zelle; die Zuteilung selbst kann bei Gleichstand abweichen). Gegenüber „alles auf einmal planen“ bringt das Aktualisieren deshalb **0, solange der Aufwand frei verteilbar ist**.
- **Im Rechteck-Muster** (je Runde ein ganz abgedecktes Rechteck; 8×8, 12 Zellen) ist kurzsichtiges Aktualisieren **nicht optimal**: 0,53 bis 6,25 Punkte hinter dem frei verteilten Optimum; mit Vorausschau über die 40 besten Rechtecke der Runde 1 (untere Schranke für das beste Rechteckpaar) 0,20 bis 2,67, im Mittel besser als kurzsichtig in 12 von 12 Zellen.
- **Alte Karte weiterbenutzen:** 0,2 bis 13,2 Punkte hinter dem aktualisierten Vorgehen (Mittel der Zellenmittel 3,5; in 23 von 24 Zellen alle 30 Instanzen schlechter). **„Dort sofort weiter bis voll“:** 0,06 bis 8,1 (Mittel 4,2).
- **Falsches α** (wahr 1, 8 Zellen = 2 Priors × 4 Budgets): Zuteilung mit α = 10 bis 16,4 Punkte hinter der mit richtigem α, mit 0,25 bis 9,7, mit 4 bis 9,0, um den Faktor 2 daneben (0,5 oder 2) 0,4 bis 2,4. Beim gleichen Faktor schadet ein zu niedriges α etwas mehr als ein zu hohes (Faktor 4: im Mittel über die 8 Zellen 7,0 gegen 5,4, in 7 von 8 Zellen; Faktor 2: 1,77 gegen 1,67, nur in 5 von 8 Zellen); der größte Schaden (16,4) entsteht erst bei der um den Faktor 10 zu hohen Annahme.

**Ansicht 3 – Weg** (5×5, Basis in der Ecke, Zufalls-Seed 1 bis 30; ein und zwei Gipfel, α ∈ {0,3; 1; 3; 10}, T ∈ {6; 10; 15}; 24 Zellen; **alle 720 CP-SAT-Lösungen bewiesen optimal**; 8×8: T ∈ {16; 32}, α ∈ {0,5; 3}, 8 Zellen, 5 von 240 Lösungen nicht bewiesen). Lücke zum Wegoptimum in Punkten, kleinster bis größter Zellenmittelwert:

| Verfahren | 5×5, ein Gipfel | 5×5, zwei Gipfel | 8×8 (beide Priors) |
|---|---|---|---|
| Wegpreis (Optimum ohne Weg minus Wegoptimum) | 0,37 bis 28,3 | 0,26 bis 35,1 | 0,57 bis 10,7 |
| Greedy kurzsichtig | 0,06 bis 4,8 | 6,4 bis 45,8 | 0,46 bis 68,2 |
| Greedy mit Ziel | 0,06 bis 3,9 | 0,34 bis 3,2 | 0,38 bis 4,4 |
| Mähen über das ganze Gebiet | 10,9 bis 49,2 | 12,0 bis 55,0 | 24,7 bis 57,4 |
| Mähen im besten Rechteck | 0,02 bis 1,1 | 0,05 bis 8,4 | 0,53 bis 14,0 |

- Mähen über das ganze Gebiet liegt in allen 24 (5×5) und 8 (8×8) Zellen mit dem Intervall über 0 (ein oder zwei Gipfel; bei flachem Prior ist es optimal, siehe unten). Im besten Rechteck (nach dem wahren Wert gewählt, mit optimierter Anfahrt: der günstigste Fall für das Verfahren, nur für diese Gitter und diese α und T) ist das Mähen bei **einem** Gipfel nahe am Optimum (höchstens 1,1 bei 5×5, 1,3 bei 8×8), bei **zwei** Gipfeln nicht überall (5×5: 0,05 bis 8,4; 8×8: bis 14,0); vermutlich weil ein Rechteck zwei getrennte Wolken nicht abdeckt (nicht eigens getestet). Eine Aussage über den Planer insgesamt ist das nicht, in der Praxis setzt man bei mehreren Sucheinheiten mehrere Rechtecke (nicht gemessen).
- **Orienteering-Fassung** (jede Zelle nur beim ersten Besuch gewertet, günstigster Fall: unter allen Orienteering-Optima der Weg mit dem höchsten wahren Wert): bei α ≤ 1 und T ≥ 10 liegt sie 4,4 bis 21,3 Punkte hinter dem Wegoptimum (8 Zellen), bei α ≤ 1 und T = 6 höchstens 3,6; bei α = 10 ist die Lücke in allen 6 Zellen gleich 0,00. Sie überschätzt die Entdeckungswahrscheinlichkeit bei α ≤ 1 um 20,2 bis 73,0 Punkte. Das ist ein **Modellunterschied** (hier sucht das Team in jedem Schritt, die Anfahrt zählt als Suchen), kein Widerspruch zur Literatur und kein Vorwurf an ein Orienteering-Verfahren.
- **Flacher Prior:** jeder Weg über T verschiedene Zellen ist optimal (T ≤ 25); Mähen über das ganze Gebiet ist dann optimal.

**Ansicht 4 – Drift** (Modell erfunden, einheitenlos; s₀ = 1, D = 0,5, sU = 0,3 heißt „gemischt“):

- **Fläche ∝ Zeit gilt nur für reine Diffusion (s₀ = 0, sU = 0);** im gemischten Beispiel ist sie mit Exponent 0,94 nahe daran, bei überwiegender Geschwindigkeitsunsicherheit wächst sie wie t². Fläche des 90-%-Bereichs relativ zu t = 1 bei t = 2 / 4 / 8: nur Diffusion 2 / 4 / 8 (Exponent 1,00); nur konstante Geschwindigkeitsunsicherheit (s₀ = D = 0) 4 / 16 / 64 (Exponent 2,00); gemischt 1,61 / 3,08 / 7,06 (Exponent 0,94 zwischen t = 1 und 8). In 30 erfundenen Zwei-Hypothesen-Szenarien (Zählung auf Zellen, 40 000 Teilchen) lag der Flächenexponent zwischen t = 1 und 4 bei 1,07 [1,01; 1,14] (Spanne 0,74 bis 1,47) und zwischen t = 4 und 8 bei 1,41 [1,36; 1,47] (Spanne 1,13 bis 1,75).
- **Geschlossene Form:** P = 1 − (1 + L)e^(−L) gleich dem Raster (im Test auf 2·10⁻³ bei Zellbreite σ/24 geprüft, das ist die Testtoleranz, nicht die gemessene Abweichung), dem Radialintegral (Testtoleranz 1·10⁻⁸) und, für α = 1, der Formel bei Hong 2022 (Abschnitt nicht am Original geprüft).
- **Verzögerungsverlust (gemischt, eine Wolke, Start t₀ = 1, Verzögerung 10 / 25 / 50 / 100 %):** bei L₀ = 0,5 4,6 / 10,7 / 19,6 / 33,4 % von P; bei L₀ = 1 3,8 / 9,0 / 16,7 / 29,2 %. Mit zwei Hypothesen (dv = 0,8, L₀ = 1) 5,3 / 12,5 / 23,1 / 39,5 %, mit dv = 0,3 4,0 / 9,6 / 17,8 / 30,9 %. Die Elastizität bei 25 % Verzögerung liegt in allen 9 gemessenen Zeilen (3 Aufwandsstufen × eine Wolke und zwei Hypothesen) unter 1, bei 0,28 bis 0,72. Der Verlust ist in diesen Zeilen **unterproportional**, nicht überproportional (gemessen als Elastizität: P fällt prozentual langsamer, als die Startzeit steigt); bei knappem Aufwand und nur unsicherem Strom (s₀ = 0,2, D = 0, sU = 0,3, L₀ = 0,5) liegt die Elastizität bei 1,26 und kann bei reinem sU gegen 2 gehen, das ist der Randfall. Die 9 Zeilen sind eine kleine, selbst gewählte Auswahl.

### Zahl → Test

| Zahl oder Aussage | Beleg |
|---|---|
| Handwerte der 3×3-Beispiele (59,25 %, 24 310 Zuteilungen, 58,5 %, 56,3125 %, 40,75 %, Wegpreis 2,0, S(t) = (1 + t/2)²) | `tests/test_search.py` (exakte Brüche) |
| Aufzählung, Wasserstand, Rucksack, Posterior, additives Prinzip, Rechteck, Schwelle und Band | `tests/test_oracle_suchaufwand.py` |
| 120 Fälle der Aufzählung, Schwelle 1080 von 1080, Band, 911 von 1080 | `data/sua_messreihe.json`, `tests/test_claims.py` (liest die Datei; einzelne Zellen im Band neu gerechnet) |
| Tabellen der Ansicht 1 und 2 | `tests/test_claims.py` (aus `data/sua_messreihe.json`; Zellen im Band 0,05 Punkte neu gerechnet) |
| Wege, Tabelle der Ansicht 3 | `tests/test_oracle_weg.py`, `tests/test_claims.py` (aus `data/sua_weg.json`; Wege auf Gültigkeit und Wert geprüft, CP-SAT nicht neu gerechnet) |
| Drift: Formeln, Verlust, Elastizität | `tests/test_search.py`, `tests/test_oracle_drift.py`, `tests/test_claims.py` |
| Drift: 30 Szenarien | `data/sua_drift.json`, `tests/test_claims.py` |

## Befunde und Korrekturen gegenüber der ersten Rechnung

Eine frühere Messung (anderer Zufallsgenerator, anderes Pairing der Instanzen) gab die Erwartungen; die Zahlen oben sind die Neu-Messung mit dem SplitMix64 der Demo. Abweichungen:

- „Vollständig“ 6,5 bis 8,9 statt 6,3 bis 9,2 (α von 0,2 bis 2), „ein Durchgang“ 25,4 bis 4,6; bei α = 10 liegt die Lücke bis 0,008 Punkte, nicht unter 0,005.
- Gelände nur 8×8 (die frühere Messung hatte auch 12×12): Vorteil von Prior je Aufwand 1,1 bzw. 2,0 statt 1,2 bzw. 2,2; Rucksack-Lücke des Greedy in 1 statt 5 von 30 kleinen Instanzen, größte Lücke 0,47 statt 4,96.
- Nichtfund: alte Karte bis 13,2 statt 13,5, falsches α = 10 bis 16,4 statt 17,9; Rechteck-Muster kurzsichtig 0,53 bis 6,25 statt 0,41 bis 5,52.
- Posterior über besuchter Zelle in 911 statt 876 von 1080 Instanzen.
- Drift-Szenarien 1,07 und 1,41 statt 1,10 und 1,48.
- Gestrichen gegenüber der ersten Rechnung: die Aussage „alle abgesuchten Zellen stehen im Posterior gleich hoch“ gilt nur bei stetigem Aufwand (bei ganzen Durchgängen gilt die Schwelle auf dem Grenzzuwachs); „Aktualisieren bringt 0“ gilt nur bei frei verteilbarem Aufwand; „Diffusion allein beschreibt Seedrift schlecht“ ist keine Aussage der Quelle und steht nicht in der Demo; „Verzögerung kostet überproportional“ ist widerlegt (unterproportional).
- Der Weg mit 25 Schritten (5×5) und die Wolken mit Seitenwechsel sind nicht gemessen.

## Ehrliche Grenzen

- **Alles erfunden.** Karten, Instanzen, Zeitmaße und Parameter sind erfunden; keine Feldversuchsdaten, keine Einsatzempfehlung.
- **Entdeckungsfunktion** 1 − e^(−α z): Zufallssuche mit ganzen Durchgängen und einer Stärke α je Durchgang. Andere Sensorformen (feste Reichweite, inverse Würfelfunktion) sind **nicht gemessen**; die Aussagen gelten nicht für jede Sensorform.
- **Ruhendes Ziel:** der Ort des Ziels ändert sich während der Suche nicht; bewegte Ziele, zerreißende Verteilungen, mehrere Teams, Hubschrauber und Sensoren mit Reichweite sind nicht Teil der Rechnung.
- **Zähleinheit** der Lücke: Prozentpunkte Entdeckungswahrscheinlichkeit; Mittel über 30 erfundene Instanzen je Zelle mit 95-%-Bootstrap-Intervall der Mittelwerte.
- **Optimum ganzzahlig** gilt bei gleichem Preis je Zelle und ist durch Aufzählung belegt, nicht aus einer gelesenen Quelle; für stetigen Aufwand ist die Zuteilung auf gleichen Grenzwert die Optimalitätsbedingung (Hong 2022 und Nakai 1987; Abschnitt bei Hong nicht am Original geprüft).
- **„Aktualisieren bringt 0“** gilt im Wert (derselbe Gesamtwert) nur bei frei verteilbarem Aufwand und gleichem Preis je Zelle; im Rechteck-Muster nicht (siehe oben).
- **Weg:** 5×5, Basis in der Ecke, kein Rückweg; Anfahrt zählt als Suchen. Bei getrennten Bewegungs- und Suchschritten gilt ein anderes Modell, in dem es einen optimalen Weg ohne Rückkehr zu einem Punkt gibt (Chakraborty u. a. 2024); das ist hier nicht gemessen. Die Maximierung der Entdeckungswahrscheinlichkeit auf Wegen ist in allgemeinen Graphen NP-schwer (Trummel und Weisinger, nach Chakraborty u. a. 2024); für das Gitter wird nichts behauptet.
- **Drift:** Gauß-Wolke, konstantes Strömungsfeld, isotrope Streuung, **feste Seite der Querwind-Hypothesen** (Seitenwechsel füllen in Wirklichkeit die Lücke zwischen den Wolken; nicht gemessen). Der Aufwand ist teilbar, die Verluste sind relativ zu P bei pünktlichem Start.
- **Lawine, Seenot mit zählendem Zeitverlauf:** zählt ein früher Fund mehr als ein später, ändert das in dieser Rechnung (ruhendes Ziel, dessen Zustand sich nicht ändert, ohne Weg) die Reihenfolge der Durchgänge, nicht die abgesuchten Zellen; ändert sich der Zustand des Ziels, ändert sich auch, welche Gebiete zuerst abgesucht werden. Eine Zahl dazu steht hier nicht.
- **Quellen** sind nur so zitiert, wie sie gelesen wurden: Frost 1996, Frost und Stone 2001, Kratzke, Stone und Frost 2010, Breivik und Allen 2008, Washburn 1998 und 2006, Hong 2022, Chakraborty u. a. 2024 und Nakai 1987 im Wortlaut (die genannten Abschnitte); Koopman, Stone, Charnes und Cooper sowie Trummel und Weisinger nur nach diesen Quellen. Die Demo behauptet keine Neuheit gegenüber der Literatur; Aussagen der Quellen sind in eigenen Worten wiedergegeben.

## Bewusst nicht umgesetzt

Weg mit Totzeit (getrennte Bewegungs- und Suchschritte), Sensor mit Reichweite, mehrere Teams, Hubschrauber, bewegtes Ziel während der Suche, zerreißende Verteilungen, nicht exponentielle Entdeckungsfunktionen, Gipfel im schwierigen Gelände, Weg im Gelände, Weg auf 12×12, Lawine (nur der Hinweis ohne Zahl), Leeway-Parameter aus Feldversuchen, Seitenwechsel der Querwind-Hypothesen, ein Löser-Vergleich für die Orienteering-Fassung.

## Verwandte Demos

- [gewinn-touren-demo](https://github.com/sebastian-hanisch/gewinn-touren-demo): Team Orienteering mit Prämie je Kunde **einmalig**. Das entspricht in der Wertung (Prämie je Zelle einmalig) der Orienteering-Fassung in Ansicht 3; Fahrzeuge, Rückkehr zum Depot und Fahrzeiten dort sind andere. Im Grenzfall großer α (jede Zelle zählt praktisch nur einmal) nähern sich beide an: bei α = 10 ist die Lücke der Fassung 0,00, bei α = 3 überschätzt sie noch; hier neu sind mehrere Durchgänge in derselben Zelle, die Gitterbewegung und die Gegenprobe Mähen. Löser (Lokalsuche, ILS, OR-Tools) sind dort, nicht hier.
- Die ganzzahlige Zuteilung ist bei gleichem Preis je Zelle separabel konkav, der Greedy ist dort exakt (durch Aufzählung belegt), nicht nur eine Näherung.
- [lagrange-kkt-demo](https://github.com/sebastian-hanisch/lagrange-kkt-demo): die Wasserstand-Lösung des teilbaren Falls ist ein Kuhn-Tucker-Punkt.
- [aufloesung-messorte-demo](https://github.com/sebastian-hanisch/aufloesung-messorte-demo): wo soll man messen statt wo soll man suchen.
- [particle-filter-demo](https://github.com/sebastian-hanisch/particle-filter-demo): die Technik der Driftwolke aus Teilchen.
- [bayes-opt-demo](https://github.com/sebastian-hanisch/bayes-opt-demo): auch dort „Posterior, dann nächster Versuch“, aber andere Beobachtung und andere Zielgröße.

## Verifikation

- **Handwerte:** 3×3-Beispiele in exakten Brüchen (`tests/test_search.py`).
- **Orakel:** Zuteilung gegen Aufzählung aller Zuteilungen, Wasserstand gegen Kuhn-Tucker, SLSQP und feinen Greedy, Rucksack gegen rekursive Aufzählung, Posterior gegen direkte Summe und Monte-Carlo, additives Prinzip, Rechteck-Baustein und Vorausschau gegen Aufzählung aller Rechteckpaare (`tests/test_oracle_suchaufwand.py`); CP-SAT gegen Aufzählung aller Wege, Orienteering-Fassung gegen Aufzählung, eingefrorene Wege (`tests/test_oracle_weg.py`); Teilchen gegen Varianzformel, geschlossene Form gegen Raster, Radialintegral und Literaturwert (`tests/test_oracle_drift.py`).
- **Eigenschaften und Zweige:** `tests/test_eigenschaften_suchaufwand.py`; **Randfälle:** `tests/test_edge_cases.py`; **Verdikte** an handgesetzten Zeilen: `tests/test_verdicts.py`.
- **Claims:** die Ergebniszahlen der Befunde stehen in `tests/test_claims.py` (Ausnahmen: Testtoleranzen und die Zahlen der früheren Messung); die Tests prüfen die Zahlen, nicht jede Formulierung darum; **Zufall:** `tests/test_zufall.py`.
- **Oberfläche:** AppTest-Rauchtest (jede Ansicht, jedes Preset, jeder Regler an den Grenzen; die Vorlage mit „jeder Slider“ passt nicht, weil die Startansicht keinen st.slider hat), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, Hervorhebung; jede Änderung eines Reglers hebt sie auf), Footer, Laufzeit-Importe, Texte.
- 253 Tests insgesamt (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kopf, Umschalter, Seitenleiste |
| `data/` | eingefrorene Messreihen, CP-SAT-Wege und Drift-Szenarien (JSON) |
| `requirements-dev.txt` | Laufzeit plus pytest, SciPy und OR-Tools (nur für Tests und Werkzeuge) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `sua_constants.py` | Ansichten, Reglerwerte, Farben, Presets |
| `sua_drift.py` | Ansicht 4: Varianz, geschlossene Form, Verlust, Raster, Teilchenwolke |
| `sua_evaluation.py` | Rechnung je Ansicht, Aggregate der Messreihen, Verdikte |
| `sua_frozen.py` | Lesezugriff auf `data/` |
| `sua_layout.py` | Fester Kopf (Stylesheet) |
| `sua_messung.py` | Zellen der Messreihen (30 Instanzen), Aufzählung |
| `sua_path.py` | Ansicht 3: Wege, Greedy, Mähmuster |
| `sua_pdf_export.py` | PDF-Export je Ansicht |
| `sua_presets.py` | Reglerangaben, Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) |
| `sua_rng.py` | SplitMix64 (Kopie der geprüften Vorlage) |
| `sua_search.py` | Kern der Ansichten 1 und 2: Prior, Gelände, Zuteilung, Posterior, Nichtfund, Rechteck |
| `sua_texts.py` | feste Texte der App |
| `sua_views.py` | die vier Ansichten (Streamlit) |
| `sua_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `sua_zufall.py` | Zufall in Blöcken auf dem SplitMix64, Bootstrap |
| `tools/sua_cpsat.py`, `tools/sua_make_weg.py` | CP-SAT-Modelle und das Werkzeug, das die Wege vorrechnet (`--check`) |
| `tools/sua_make_zuteilung.py`, `tools/sua_make_drift.py` | Werkzeuge für die Messreihen und die Drift-Szenarien (`--check`) |
| `tests/` | Tests (siehe Verifikation) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/sua_make_zuteilung.py --check
python tools/sua_make_drift.py --check
python tools/sua_make_weg.py --check
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2; Tests und Werkzeuge nutzen SciPy und OR-Tools.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
