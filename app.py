"""Suchaufwand: wo sucht man, wenn nichts gefunden wurde - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Ansichten (Umschalter): Zuteilung und Gelände, Nichtfund und falsches α, Weg, Drift und Verzögerung. Anwendungsbeispiel, keine Beratung;
alle Instanzen erfunden. Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sua_constants as C
import sua_texts as T
import sua_views as V
from sua_layout import dunkles_design, kopf_css
from sua_presets import init_session_state_defaults, load_permalink_settings, sidebar_presets, sync_query_params

st.set_page_config(page_title="Suchaufwand – Sebastian Hanisch", layout="wide")

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sua_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 " + T.TITEL)
st.markdown(T.EINLEITUNG)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/suchaufwand-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Team sucht in Gelände oder auf See nach einem Vermissten. Wo er ist, weiß niemand genau, nur wie wahrscheinlich er an welcher Stelle ist: Das ist die Karte. Ein Durchgang durch ein Feld findet ihn, falls er dort ist, nur mit einer gewissen Chance. Die Demo fragt in vier Ansichten: Wohin gehört der nächste Durchgang? Was ändert ein Nichtfund an der Karte? Was kostet der Weg des Teams? Und wie wächst die Wolke möglicher Orte, wenn ein Ziel treibt und später gesucht wird?

Alles ist erfunden und stark vereinfacht. Die Demo ist ein Anwendungsbeispiel, keine Beratung und keine Einsatzempfehlung.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/suchaufwand-erklaert.html)
        """
    )
# erklaerseite:einfach:end

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben

ansicht = st.radio("Ansicht", C.ANSICHTEN, key="ansicht", format_func=C.ANSICHT_NAMEN.get, horizontal=True)
V.ANSICHTEN[ansicht]()
sync_query_params(ansicht)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
