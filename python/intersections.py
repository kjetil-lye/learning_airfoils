def find_minimum(measurements):
    best_values = {}
    best_configurations = {}
    for m in measurements:
        for k in m.keys():
            if 'speedup' in k:
                continue
            if k not in best_values or best_values[k] > m[k]:
                best_values[k] = m[k]
                best_configurations[k] = copy.deepcopy(m.properties)
    return best_values, best_configurations


def find_maximum(measurements):
    best_values = {}
    best_configurations = {}
    for m in measurements:
        for k in m.keys():
            if 'speedup' in k:
                continue
            if k not in best_values or best_values[k] < m[k]:
                best_values[k] = m[k]
                best_configurations[k] = copy.deepcopy(m.properties)
    return best_values, best_configurations


def find_close_configurations(measurements, key_to_compare, value_of_key, distance):
    compatible_measurements = []
    for m in measurements:
        value = m[key_to_compare]
        if abs(value-value_of_key) <= distance:
            compatible_measurements.append(m)
    return compatible_measurements
