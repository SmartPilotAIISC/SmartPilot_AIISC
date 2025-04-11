
import json
from collections import defaultdict

class ProcessOntologyQA:
    def __init__(self, ontology_path):
        with open(ontology_path, "r") as f:
            self.ontology = json.load(f)
        self.nodes = self.ontology.get("nodes", [])
        self.links = self.ontology.get("links", [])
        self.node_map = {n["id"]: n for n in self.nodes}
        self.name_to_id = {n.get("item_name", "").lower(): n["id"] for n in self.nodes if "item_name" in n}

        # Build adjacency map
        self.adjacency = defaultdict(list)
        for link in self.links:
            self.adjacency[link["source"]].append((link["target"], link["relationship"]))
            self.adjacency[link["target"]].append((link["source"], link["relationship"]))

    def get_ontology_data(self):
        """Return the full ontology JSON (nodes and links)."""
        return self.ontology

    def answer_query(self, query: str):
        q = query.lower()

        if "types of sensors" in q or "what sensors" in q:
            return self.get_sensor_types()

        if "details on the sensors" in q or "more details" in q:
            return self.get_all_sensor_details()

        if "why did the anomaly" in q:
            import re
            match = re.search(r"why did the anomaly ([\w\d_]+)", q)
            if match:
                anomaly = match.group(1)
                return self.get_anomaly_reason(anomaly)

        if "anomalies associated with sensor" in q:
            import re
            match = re.search(r"anomal(?:y|ies) associated with sensor (\w+)", q)
            if match:
                sensor_name = match.group(1)
                return self.get_anomalies_for_sensor(sensor_name)

        return "🤔 Sorry, I couldn't interpret that query."

    def get_sensor_types(self):
        sensor_types = {n.get("type") for n in self.nodes if n["type"].lower() == "sensor"}
        return sorted(sensor_types) if sensor_types else ["No sensor types found."]

    def get_all_sensor_details(self):
        sensor_details = []
        for n in self.nodes:
            if n["type"].lower() == "sensor":
                detail = f"🔧 {n.get('item_name', 'Unknown')} - {n.get('measures', 'No description')} (Spec: {n.get('item_spec', 'N/A')})"
                sensor_details.append(detail)
        return sensor_details if sensor_details else ["No sensors found."]

    def get_anomalies_for_sensor(self, sensor_name):
        sensor_id = self.name_to_id.get(sensor_name.lower())
        if not sensor_id:
            return [f"❌ Sensor `{sensor_name}` not found."]

        visited = set()
        queue = [sensor_id]
        anomaly_types = set()

        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            node = self.node_map.get(current)
            if node and node["type"] == "Cycle":
                anomaly_types.update(node.get("anomaly_types", []))
            queue.extend([target for target, _ in self.adjacency[current]])

        if anomaly_types:
            return [f"🚨 Anomaly types related to `{sensor_name}`:"] + list(anomaly_types)
        else:
            return [f"ℹ️ No anomalies directly associated with `{sensor_name}`."]

    def get_anomaly_reason(self, anomaly_name):
        anomaly_name = anomaly_name.lower()
        related_cycles = []

        for node in self.nodes:
            if node["type"] == "Cycle":
                for a in node.get("anomaly_types", []):
                    if a.lower() == anomaly_name:
                        cycle_number = node.get("cycle_state", "Unknown")
                        cycle_func = node.get("cycle_function", "No description")
                        related_cycles.append(f"🔁 Cycle {cycle_number} – {cycle_func}")

        if related_cycles:
            return [f"📍 The anomaly `{anomaly_name}` is associated with the following cycle(s):"] + related_cycles
        else:
            return [f"❓ No information available for anomaly `{anomaly_name}`."]

    def get_robot_function(self, robot_name):
        robot_id = self.name_to_id.get(robot_name.lower())
        if not robot_id:
            return f"❌ Robot `{robot_name}` not found."
        robot_node = self.node_map.get(robot_id, {})
        return f"🤖 `{robot_name}` function: {robot_node.get('function', 'No function described.')}"

    def get_gripper_of_robot(self, robot_name):
        robot_id = self.name_to_id.get(robot_name.lower())
        if not robot_id:
            return f"❌ Robot `{robot_name}` not found."
        for neighbor_id, rel in self.adjacency[robot_id]:
            if rel == "has" and self.node_map[neighbor_id]["type"] == "Gripper":
                return f"🦾 `{robot_name}` uses `{self.node_map[neighbor_id]['item_name']}`"
        return f"ℹ️ No gripper found for robot `{robot_name}`."

    def get_sensors_of_robot(self, robot_name):
        robot_id = self.name_to_id.get(robot_name.lower())
        if not robot_id:
            return [f"❌ Robot `{robot_name}` not found."]
        sensors = set()
        for n1, rel1 in self.adjacency[robot_id]:
            if rel1 == "has" and self.node_map[n1]["type"] == "Gripper":
                for n2, rel2 in self.adjacency[n1]:
                    if rel2 == "has_sensor" and self.node_map[n2]["type"] == "Sensor":
                        sensors.add(self.node_map[n2]["item_name"])
        return list(sensors) if sensors else [f"ℹ️ No sensors found for `{robot_name}`."]

    def get_sensor_description(self, sensor_name):
        sensor_name_key = sensor_name.lower().strip()
        sensor_id = self.name_to_id.get(sensor_name_key)

        if not sensor_id:
            # Try fuzzy match fallback
            for name in self.name_to_id:
                if sensor_name_key in name:
                    sensor_id = self.name_to_id[name]
                    break

        if not sensor_id:
            return None  # 🔄 Let the app fallback to LLM

        node = self.node_map[sensor_id]
        node_type = node.get("type", "")

        if node_type == "Sensor":
            return (
                f"📡 Sensor `{sensor_name}`:\n"
                f"- Measures: {node.get('measures', 'N/A')}\n"
                f"- Spec: {node.get('item_spec', 'N/A')}"
            )
        elif node_type == "Sensor_Value":
            return (
                f"📈 Sensor Value `{sensor_name}`:\n"
                f"- Description: {node.get('description', 'No description')}\n"
                f"- Min: {node.get('min_value', 'N/A')}, Max: {node.get('max_value', 'N/A')}\n"
                f"- Unit: {node.get('unit', 'N/A')}"
            )
        else:
            return None

    def get_sensor_value_range(self, sensor_value_name):
        value_id = self.name_to_id.get(sensor_value_name.lower())
        if not value_id:
            return f"❌ Sensor value `{sensor_value_name}` not found."
        node = self.node_map[value_id]
        return (
            f"📏 `{sensor_value_name}` range: {node.get('min_value', 'NA')} to {node.get('max_value', 'NA')} "
            f"(Unit: {node.get('unit', 'N/A')})"
        )

    def trace_robot_to_sensor_value(self, robot_name):
        robot_id = self.name_to_id.get(robot_name.lower())
        if not robot_id:
            return [f"❌ Robot `{robot_name}` not found."]
        trace_paths = []
        for n1, rel1 in self.adjacency[robot_id]:
            if rel1 == "has" and self.node_map[n1]["type"] == "Gripper":
                gripper_name = self.node_map[n1]["item_name"]
                for n2, rel2 in self.adjacency[n1]:
                    if rel2 == "has_sensor":
                        sensor_name = self.node_map[n2]["item_name"]
                        for n3, rel3 in self.adjacency[n2]:
                            if rel3 == "has_value":
                                value_name = self.node_map[n3]["item_name"]
                                trace_paths.append(f"{robot_name} ➝ {gripper_name} ➝ {sensor_name} ➝ {value_name}")
        return trace_paths if trace_paths else [f"ℹ️ No sensor values traced for `{robot_name}`."]
