## Starten mit abaqus cae noGUI=auswertung.py
## Das Skript durchlaeuft alle .odb-files in einem Ordner und schreibt die extrahierten Ergebnisse in die Datei models_list_results.json (zusammen mit den Daten aus models_list.json)

import abaqusConstants as abqC
import odbAccess
import glob
import re
import json
import numpy as np
import os

odb_dir_path = "../../finished_abaqus_files_and_param_fields/param_field_1/odb-files/2022"
os.chdir(odb_dir_path)
models_list_path = "C:/Projekte/datengenerierung/finished_abaqus_files_and_param_fields/param_field_1/models_list.json"
result_file_path = "C:/Projekte/datengenerierung/finished_abaqus_files_and_param_fields/param_field_1/result-files/models_list_results.json"

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    if isinstance(data, str):
        return data

    # If this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # If this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.items() # changed to .items() for Python 2.7/3
        }

    # Python 3 compatible duck-typing
    # If this is a Unicode string, return its string representation
    if str(type(data)) == "<type 'unicode'>":
        return data.encode('utf-8')

    # If it's anything else, return it in its original form
    return data

odb_dict = {re.match("abaqus_(\d+)_(\d+)_([a-zA-Z0-9]+)\.odb", fname).group(3): {"odb_file": fname} for fname in glob.glob('*.odb')}
# odb_dict = {re.match("([a-zA-Z0-9]+)\.odb", fname).group(1): {"odb_file": fname} for fname in glob.glob('*.odb')}

def getMaxInNodeset(odb, nodeset, invariant=None, component=None):
    if not invariant and not component:
        raise ValueError('Only one of invariant and component can be specified.')
    
    if component:
        data = [(v.nodeLabel, v.data) for v in odb.steps['Step-1'].frames[1].fieldOutputs['S'].getSubset(position=abqC.ELEMENT_NODAL).getSubset(region=nodeset).getScalarField(componentLabel=component).values]
    elif invariant:
        data = [(v.nodeLabel, v.data) for v in odb.steps['Step-1'].frames[1].fieldOutputs['S'].getSubset(position=abqC.ELEMENT_NODAL).getSubset(region=nodeset).getScalarField(invariant=invariant).values]
    data_dict = {}
    for label, val in data:
        if not label in data_dict:
            data_dict[label] = [val]
        else:
            data_dict[label].append(val)
    data_dict = {label: sum(vlist)/len(vlist) for label, vlist in data_dict.items()}
    max_nodelabel = max(data_dict, key=data_dict.get)

    tensors = []
    for v in odb.steps['Step-1'].frames[1].fieldOutputs['S'].getSubset(position=abqC.ELEMENT_NODAL).getSubset(region=nodeset).values:
        if v.nodeLabel == max_nodelabel:
            tensors.append(np.array(v.data))
    tensor = np.array(tensors).mean(axis=0)

    for node in nodeset.nodes:
        if node.label == max_nodelabel:
            coordinates = list(node.coordinates)
            break
    
    return max_nodelabel, data_dict[max_nodelabel], coordinates, tensor


for key in odb_dict:
    odb = odbAccess.openOdb(path=odb_dict[key]['odb_file'])
    ## EVAL_POINT
    p_nodeLabel, p_value, p_coordinates, p_tensor = getMaxInNodeset(odb, odb.rootAssembly.instances['FILLET_WELD-1'].nodeSets['EVAL_POINT'], invariant=abqC.MAX_PRINCIPAL)
    ## EVAL_RADIUS
    r_nodeLabel, r_value, r_coordinates, r_tensor = getMaxInNodeset(odb, odb.rootAssembly.instances['FILLET_WELD-1'].nodeSets['EVAL_RADIUS'], invariant=abqC.MAX_PRINCIPAL)
    odb_dict[key].update(
        {
            "results": {
                "eval_radius": {
                    "S_MAX_PRINCIPAL": {
                        "value": r_value,
                        "position": [float(x) for x in r_coordinates],
                        "nodeLabel": r_nodeLabel,
                        "tensor": [float(x) for x in r_tensor]
                    }
                },
                "eval_point": {
                    "S_MAX_PRINCIPAL": {
                        "value": p_value,
                        "position": [float(x) for x in p_coordinates],
                        "nodeLabel": p_nodeLabel,
                        "tensor": [float(x) for x in p_tensor]
                    }
                }
            }
        }
    )

    odb.close()

with open(models_list_path, 'r') as f:
    model_json = json_load_byteified(f)
for key, value in odb_dict.items():
    model_json[key].update(value)
# print(model_json)
with open(result_file_path, 'w') as f:
    json.dump(model_json, f, indent=4)
