# Benutzung

In der Datei `param_field.json` die Parameter-Sets festlegen. Danach mit folgendem Befehl die Erstellung der Input-Files durchführen:

```powershell
abaqus cae noGUI=generate_models.py
```

Für jeden job wird ein `.cae`, ein `.jnl` und ein `.inp` file erstellt. Die Namen sind der md5-hash des folgenden Strings:

```python
'bending_%6.8f_%6.8f_%6.8f_%6.8f' % (r, t, w, alpha)
```

Weiterhin wird die Datei `models_list.json` erstellt, die ein dict mit den md5-hashes sowie den Parametern enthält.

Um Job zu starten:
```powershell
abaqus job=$jobname interactive
```

Um die Kerbformzahlen der odb-Dateien auszuwerten:
```powershell
abaqus cae noGUI=generate_models.py
```
Dieses Skript generiert eine `models_list_results.json` Datei mit den ausgewerteten Werten für alle odb-Dateien.