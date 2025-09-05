# Datengenerierung
Dieser Ordner beinhaltet die Skripte zur Generierung von künstlichen Stumpf- und Kreuzstößen.

Mit Hilfe von `generate_parameter_fields.py` kann man Geometrieparameter für die jeweiligen Schweißverbindungstypen definieren und eine json-Datei erstellen, welche alle möglichen Parameterkombinationen beinhaltet.

Mit den Skripts in `data_generation/generate_abaqus_models` lassen sich Abaqus-Modelle aus dem erzeugten Parameterfeld erstellen. 

Mit `data_generation/point_cloud_generierung/stumpfstoß_pointcloud.py` lassen sich Punktwolken von Stumpfstößen erzeugen. Die Erzeugung der Punktwolken für Kreuzstöße geschieht anhand der Abaqus-Modelle in `data_generation/generate_abaqus_models/fillet_weld/generate_models.py`. 

# Erzeugte Daten
Im Rahmen dieser Masterarbeit wurden künstliche Daten von Stumpfstößen und Kreuzstößen erzeugt.
In `finished_abaqus_files_and_param_fields` befinden sich für die Stumpfstöße (param_field_0) und Kreuzstöße (param_field_1):
- Die `param_field_butt_weld.json` bzw `param_field_fillet_weld.json` mit den definierten Kombinationen an Parametern, welche zur Erzeugung der Abaqus-Modelle und Pointclouds genutzt wurden
- Die Datei `models_list.json` mit den Geometrieparametern für jedes Schweißmodell
- die result-file `models_list_results.json` mit den Geometrieparametern und den ermittelten Kerbformzahlen aus den Abaqus-Simulationen für jedes Schweißmodell
- die generierten Pointclouds
