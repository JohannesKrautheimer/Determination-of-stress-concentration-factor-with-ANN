# Andere Methoden

In diesem Ordner befindet sich die Implementierung der Approximationsformeln nach Anthes, Rainer & Kiyak und die Methode basierend auf KNN von Oswald.

## Oswald
Um die Kerbformzahl nach Oswald zu bestimmen (`get_oswald_scf_from_website` in `scf_approximation_formulas.py`) wird ein internet driver benötigt. Die .exe dieses drivers wird im [resources](resources) Ordner abgelegt. Je nach Browsertyp muss die Codezeile `driver = webdriver.Edge...`in der `prepare_internet_driver`-Methode in `scf_approximation_formulas.py` auf den entsprechenden Browser geändert werden.