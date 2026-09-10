# File vuoto necessario perche' pytest, in modalita' di import di default,
# aggiunge a sys.path solo la cartella del conftest.py piu' vicino alla radice.
# Senza questo file, "from predictor import SentimentPredictor" in
# tests/test_smoke.py fallirebbe con ModuleNotFoundError, perche' pytest
# aggiungerebbe a sys.path solo tests/, non la radice del repository dove
# vive predictor.py.
