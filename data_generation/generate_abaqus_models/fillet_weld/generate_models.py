# coding=utf-8
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import *
from itertools import compress
import hashlib
import part
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

def check_limitations_for_reinforcement(h, a):
    limitations_passed = True
    #Limitations
    #DIN EN ISO 5817 Bewertungsgruppe C
    #Item 503 Zu große Nahtüberhöhung (Kehlnaht)
    b = 2 * a * np.tan(np.deg2rad(45))
    if not (h <= 1 + 0.15 * b and h <= 4):
        limitations_passed = False    
        print("Zu große Nahtüberhöhung (Item 503)! h <= 1 mm + 0,15 b, aber max. 4 mm")
        print("Params:")
        print("h, a:", h, a)
    #Item 5213 Zu kleine Kehlnahtdicke
    if not (abs(h) <= 0.3 + 0.1 * a and abs(h) <= 1):
        limitations_passed = False
        print("Zu kleine Kehlnahtdicke (Item 5213)! h <= 0,3 mm + 0,1 a, aber max. 1 mm")
        print("Params:")
        print("h, a:", h, a)
     #Item 5214 Zu große Kehlnahtdicke
    if not (h <= 1 + 0.2 * a and h <= 4):
        limitations_passed = False
        print("Zu große Kehlnahtdicke (Item 5214)! h <= 1 mm + 0,2 a, aber max. 4 mm")
        print("Params:")
        print("h, a:", h, a)
    return limitations_passed

def update_model(r, t, aval, alpha):
    if alpha == 135.0:
        return update_model_('model_template/filletweld_135.cae', r, t, aval, alpha)
    elif alpha < 135.0:
        return update_model_('model_template/filletweld_l135.cae', r, t, aval, alpha)
    elif alpha > 135.0:
        return update_model_('model_template/filletweld_g135.cae', r, t, aval, alpha)

# Returns -1, -1 if the limitations are not passed
def update_model_(filename_cae, r, t, aval, alpha):
    mdb_obj = openMdb(filename_cae)

    p = mdb_obj.models['Model-1'].parts['fillet_weld']

    #mdb_obj.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=p.features['Partition face-1'].sketch)
    #mdb_obj.models['Model-1'].sketches['__edit__'].parameters['w2'].setValues(expression='%6.8f' % w)
    #p.features['Partition face-1'].setValues(sketch=mdb_obj.models['Model-1'].sketches['__edit__'])
    #del mdb_obj.models['Model-1'].sketches['__edit__']

    mdb_obj.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=p.features['Shell planar-1'].sketch)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['t'].setValues(expression='%6.8f' % t)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['a'].setValues(expression='%6.8f' % aval)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['alpha'].setValues(expression='%6.8f' % alpha)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['r'].setValues(expression='%6.8f' % r)
    p.features['Shell planar-1'].setValues(sketch=mdb_obj.models['Model-1'].sketches['__edit__'])
    del mdb_obj.models['Model-1'].sketches['__edit__']

    p.regenerate()

    mdb_obj.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=p.features['Shell planar-1'].sketch)
    mdb_obj.models['Model-1'].sketches['__edit__'].parameters['t'].setValues(expression='%6.8f' % t)
    h = mdb_obj.models['Model-1'].sketches['__edit__'].parameters['h'].value
    limitations_passed = check_limitations_for_reinforcement(h, a=aval)
    if not limitations_passed:
        return -1, -1
    del mdb_obj.models['Model-1'].sketches['__edit__']

    mdb_obj.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=p.features['Partition face-1'].sketch)
    c_x, c_y, c_r = [mdb_obj.models['Model-1'].sketches['__edit__'].parameters[v].value for v in ['c_x', 'c_y', 'c_r']]
    del mdb_obj.models['Model-1'].sketches['__edit__']

    mesh_export_edge_list = list(compress(p.edges.getByBoundingBox(xMin=4, xMax=51, yMin=4, yMax=31, zMin=0, zMax=0), [sum([1 if e.index in f.getEdges() else 0 for f in p.faces]) == 1 for e in p.edges.getByBoundingBox(xMin=4, xMax=51, yMin=4, yMax=31, zMin=0, zMax=0)]))
    if 'mesh_export' in p.sets:
        del p.sets['mesh_export']
    p.Set(name='mesh_export', edges=part.EdgeArray(mesh_export_edge_list))

    if 'xsymm' in p.sets:
        del p.sets['xsymm']
    p.Set(name='xsymm', edges=p.edges.getByBoundingBox(xMin=-1, xMax=1, yMin=-31, yMax=31, zMin=0, zMax=0))

    if 'rp_set' in p.sets:
        del p.sets['rp_set']
    p.Set(name='rp_set', edges=p.edges.getByBoundingBox(xMin=49, xMax=51, yMin=-6, yMax=6, zMin=0, zMax=0))

    if 'pinned' in p.sets:
        del p.sets['pinned']
    p.Set(name='pinned', vertices=p.vertices.getByBoundingBox(xMin=0, xMax=0, yMin=0, yMax=0, zMin=0, zMax=0))

    if 'mesh_radius' in p.sets:
        del p.sets['mesh_radius']
    p.Set(name='mesh_radius', edges=p.edges.getByBoundingBox(xMin=c_x-c_r-0.01, xMax=c_x+c_r+0.01, yMin=c_y-c_r-0.01, yMax=c_y+c_r+0.01, zMin=0, zMax=0))
    
    p.regenerate()

    a = mdb_obj.models['Model-1'].rootAssembly
    a.regenerate()

    mdb_obj.models['Model-1'].loads['moment'].setValues(cm3=-(float(t)**2.0)/6.0, distributionType=UNIFORM, field='')

    p.deleteMesh()
    p.deleteSeeds(p.edges)
    p.seedEdgeBySize(edges=p.sets['mesh_export'].edges, size=0.025, deviationFactor=0.1, minSizeFactor=0.1, constraint=FINER)
    [p.generateMesh(regions=p.faces.getByBoundingBox(xMin=-1, xMax=51, yMin=-1, yMax=31, zMin=0, zMax=0)[i:i+1]) for i in range(len(p.faces.getByBoundingBox(xMin=-1, xMax=51, yMin=-1, yMax=31, zMin=0, zMax=0)))]
    node_coordinates = np.array([n.coordinates for n in p.sets['mesh_export'].nodes])
    print(node_coordinates.shape)
    
    p.deleteMesh()
    p.deleteSeeds(p.edges)
    p.seedEdgeBySize(edges=p.sets['mesh_radius'].edges, size=0.025, deviationFactor=0.1, minSizeFactor=0.1, constraint=FINER)
    p.generateMesh()
    a.regenerate()

    job_name = 'filletweld_bending_%6.8f_%6.8f_%6.8f_%6.8f' % (r, t, aval, alpha)
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
    
    np.savetxt('../../finished_abaqus_files_and_param_fields/param_field_1/point_clouds/%s.csv' % job_name_encoded, node_coordinates)
    np.save('../../finished_abaqus_files_and_param_fields/param_field_1/point_clouds/%s.npy' % job_name_encoded, node_coordinates, allow_pickle=False)

    return job_name_encoded, h


with open('param_field.json', 'r') as f:
    data = json_load_byteified(f)

result_json = {}

num_valid_configs = 0
for s in data:
    r = s['radius']
    t = s['thickness']
    throat = s['throat']
    alpha = s['alpha']
    job_name_encoded, h = update_model(r, t, throat, alpha)
    if job_name_encoded == -1 and h == -1:
        print("Skipping this param configuration:")
        print("r, t, throat, alpha: ", r, t, throat, alpha)
        print("########################")
        continue
    result_json.update({job_name_encoded: {'input_file': '%s.inp' % job_name_encoded, 'data': s , 'reinforcement': h}})
    num_valid_configs += 1

print("In total " + str(len(data)) + " param configurations were checked.")
print("Only for " + str(num_valid_configs) + " configurations an abaqus model was build.")

with open('models_list.json', 'w') as f:
    json.dump(result_json, f, indent=4)
