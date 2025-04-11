import re
import pandas as pd
from collections import defaultdict


def parse_sensor_ranges(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sensor_ranges = {}
    current_sensor = None
    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("cycle state"):
            continue

        if re.match(r'^[Ii]_?[Rr][0-9]+_?[Gg]ripper.*', line):
            current_sensor = line.strip().lower()  # ✅ Normalize sensor name
            sensor_ranges[current_sensor] = {}
        elif re.match(r'^\d+', line) and current_sensor:
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                cycle = int(parts[0])
                min_max = re.findall(r'\d+', parts[1])
                if len(min_max) == 2:
                    min_val, max_val = map(int, min_max)
                    sensor_ranges[current_sensor][cycle] = (min_val, max_val)
    return sensor_ranges


def is_anomalous(sensor, value, cycle, sensor_ranges):
    sensor = sensor.lower()  # ✅ Normalize input sensor
    if sensor not in sensor_ranges:
        return False
    if cycle not in sensor_ranges[sensor]:
        return False
    min_val, max_val = sensor_ranges[sensor][cycle]
    return not (min_val <= value <= max_val)


def analyze_row_for_root_causes(row, sensor_ranges, adj_matrix, node_labels):
    cycle = int(row['CycleState'])
    actual_state = row['actual_state']
    root_causes = []

    node_labels_lower = [n.lower() for n in node_labels]
    row_columns_lower = {col.lower(): col for col in row.index}

    for sensor in sensor_ranges:
        sensor_lc = sensor.lower()
        sensor_col = row_columns_lower.get(sensor_lc)
        if sensor_col:
            print(f"✅ Found sensor column match: {sensor_lc} → {sensor_col}")
            if pd.notna(row[sensor_col]):
                try:
                    value = float(row[sensor_col])
                    print(f"🧪 Checking {sensor_col} value={value} at cycle={cycle}")
                    if is_anomalous(sensor_lc, value, cycle, sensor_ranges):
                        print(f"❗ Anomaly detected in {sensor_col}!")
                        root_causes.append(sensor_col.lower())  # ✅ Normalize root cause name
                except Exception as e:
                    print(f"⚠️ Error parsing value for {sensor_col}: {e}")

    cause_paths = {}
    for sensor in root_causes:
        if sensor not in node_labels_lower:
            continue
        j = node_labels_lower.index(sensor)
        parents = [(node_labels[i].lower(), adj_matrix[i, j]) for i in range(len(node_labels)) if adj_matrix[i, j] != 0]
        cause_paths[sensor] = parents  # ✅ Store lowercased names

    return {
        'actual_state': actual_state,
        'CycleState': cycle,
        'anomalous_sensors': root_causes,
        'root_cause_paths': cause_paths
    }


def analyze_all(df, sensor_ranges, adj_matrix, node_labels):
    results = []
    for idx, row in df.iterrows():
        results.append(analyze_row_for_root_causes(row, sensor_ranges, adj_matrix, node_labels))
    return results


def compute_rca_statistics(rca_results):
    def normalize(x):
        return x.strip().lower()

    cause_freq = defaultdict(int)
    cause_strengths = defaultdict(float)
    total_counts = defaultdict(int)

    for result in rca_results:
        for sensor, causes in result["root_cause_paths"].items():
            sensor_norm = normalize(sensor)
            for parent, strength in causes:
                key = (normalize(parent), sensor_norm)
                cause_freq[key] += 1
                cause_strengths[key] += abs(strength)
                total_counts[sensor_norm] += 1

    avg_strengths = {k: cause_strengths[k] / cause_freq[k] for k in cause_freq}

    # Debug print
    print("📊 RCA Aggregated Keys (parent ➝ sensor):")
    for (p, s), avg in avg_strengths.items():
        print(f"  {p} ➝ {s} | Avg strength = {avg:.4f}")

    return cause_freq, avg_strengths, total_counts
