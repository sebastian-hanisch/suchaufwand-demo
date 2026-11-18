"""Eingefrorene Dateien in `data/`: Aufbau, Zellenzahlen, Verweise auf die Werkzeuge. Die Werte selbst prüfen `test_claims.py` und `test_oracle_weg.py`."""

import sua_frozen as F


def test_messreihe_hat_alle_zellen():
    d = F.laden("sua_messreihe")
    assert len(d["s1"]) == 180 and len(d["tau"]) == 72 and len(d["s2"]) == 120 and len(d["nichtfund"]) == 24 and len(d["falsches_alpha"]) == 8
    assert len(d["rechteck_runden"]) == 12 and len(d["posterior_band"]) == 36 and d["meta"]["instanzen"] == list(range(1, 31))
    for row in d["s1"].values():
        for k in ("voll", "durchgang", "rechteck"):
            assert row[k]["lo"] <= row[k]["mittel"] <= row[k]["hi"] + 1e-12 and 0 <= row[k]["anteil"] <= 1


def test_schluessel_der_app_finden_ihre_zellen():
    assert F.zelle("s1", 8, "ein_gipfel", 1.0, 1.0)["B"] == 64 and F.zelle("s1", 12, "flach", 10.0, 0.125)["B"] == 18
    assert F.zelle("s2", "zwei_gipfel", 0.5, 3.0, 2.0) is not None and F.zelle("tau", "ein_gipfel", 1.0, 0.5, 0.99) is not None
    assert F.zelle("nichtfund", "ein_gipfel", 1.0, 1.0) is not None and F.zelle("falsches_alpha", "zwei_gipfel", 0.25) is not None
    assert F.zelle("s1", 8, "ein_gipfel", 1.5, 1.0) is None and F.messreihe("gibt_es_nicht") == {}


def test_drift_und_weg_dateien_haben_ihre_kopfzeilen():
    assert F.drift_szenarien()["meta"]["szenarien"] == 30
    assert F.laden("sua_weg")["meta"]["zeit_5x5_s"] == 6.0 and F.laden("gibt_es_nicht") is None
