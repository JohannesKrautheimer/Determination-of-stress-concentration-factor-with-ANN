from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import *
import hashlib
import numpy as np

executeOnCaeStartup()


import json

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

def update_model(r, t, w, alpha):
    mdb_obj = openMdb('model_template/butt_weld_model_v2.cae')

    p = mdb_obj.models['Model-1'].parts['butt_weld']

    mdb_obj.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=p.features['Partition face-1'].sketch)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['w2'].setValues(expression='%6.8f' % w)
    p.features['Partition face-1'].setValues(sketch=mdb_obj.models['Model-1'].sketches['__edit__'])
    del mdb_obj.models['Model-1'].sketches['__edit__']

    mdb_obj.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=p.features['Shell planar-1'].sketch)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['t'].setValues(expression='%6.8f' % t)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['w'].setValues(expression='%6.8f' % w)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['alpha'].setValues(expression='%6.8f' % alpha)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['r'].setValues(expression='%6.8f' % r)
    p.features['Shell planar-1'].setValues(sketch=mdb_obj.models['Model-1'].sketches['__edit__'])
    del mdb_obj.models['Model-1'].sketches['__edit__']

    p.regenerate()

    a = mdb_obj.models['Model-1'].rootAssembly
    a.regenerate()

    mdb_obj.models['Model-1'].loads['moment'].setValues(cm3=-(float(t)**2.0)/6.0, distributionType=UNIFORM, field='')

    p.generateMesh()
    a.regenerate()

    job_name = 'bending_%6.8f_%6.8f_%6.8f_%6.8f' % (r, t, w, alpha)
    m = hashlib.md5()
    m.update(job_name.encode('utf8'))
    job_name_encoded = m.hexdigest()
    mdb_obj.Job(name=job_name_encoded, model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb_obj.jobs[job_name_encoded].writeInput(consistencyChecking=OFF)

    print('Wrote input file: %s' % job_name_encoded)

    mdb_obj.saveAs(pathName="other_files/"+str(job_name_encoded))
    mdb_obj.close()

    return job_name_encoded

with open('param_field.json', 'r') as f:
    data = json_load_byteified(f)
result_json = {}

for s in data:
    r = s['radius']
    t = s['thickness']
    w = s['width']
    alpha = s['alpha']
    #Limitation of radius for small angles. Defined by Matthias such that at least 5 elements fit on the radius
    r_min_allowed = (0.025 * 5) * (180 / np.pi) * 1 / alpha
    if r < r_min_allowed:
        print("Radius is smaller than minimum allowed radius of " + str(r_min_allowed) + "!")
        print("Params:")
        print("r, t, w, alpha, h: ", r, t, w, alpha)
        print("Skipping this param configuration...")
        print("########################")
        continue
    job_name_encoded = update_model(r, t, w, alpha)
    result_json.update({job_name_encoded: {'input_file': '%s.inp' % job_name_encoded, 'data': s }})

with open('models_list.json', 'w') as f:
    json.dump(result_json, f, indent=4)
