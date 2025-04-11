import re

class AssetLoader:

    @staticmethod
    def get_queries():
        documentation_queries = [
            "How to set up the toy rocket manufacturing machine?",
            "What are the safety protocols for the manufacturing process?",
            "How to troubleshoot common issues in the manufacturing pipeline?",
            "Describe the maintenance procedure for the assembly line machines.",
            "What are the steps to calibrate the sensors in the manufacturing setup?",
            "How to perform a quality check on the manufactured toy rockets?",
            "What materials are needed for the manufacturing process?",
            "How to store and handle materials safely?",
            "What are the emergency procedures in case of a malfunction?",
            "How to document the production cycle for future reference?"
        ]

        anomaly_prediction_queries = [
            "Enter the current sensor values to check if an anomaly will happen next",
        ]

        prod_forecast_queries = [
            "Enter the current production statistics to get the next production",
        ]

        causal_agent_queries=[
            "Upload your .csv dataset to analyse causal discovery"
        ],

        queries = {
            "Documentation": documentation_queries,
            "Anomaly Prediction and Sensor Values": anomaly_prediction_queries,
            "Production Forecasting": prod_forecast_queries,
            "Causal analysis": causal_agent_queries
        }
        return queries

    @staticmethod
    def get_templates():
        system_templates = {
            "Documentation": """You assist users with tasks related to the documentation of the toy rocket manufacturing pipeline. Your responsibilities include:
                                - Providing detailed instructions on setting up and operating manufacturing machinery.
                                - Explaining safety protocols and emergency procedures.
                                - Guiding users through troubleshooting common issues.
                                - Describing maintenance and calibration procedures.
                                - Assisting with quality checks and documentation of the production process.
                                - Ensuring users understand material handling and storage protocols.""",
            
            "Anomaly Prediction and Sensor Values": """You assist users with tasks related to predicting anomalies and determining sensor values in the toy rocket manufacturing pipeline. Your responsibilities include:
                                                       - Predicting anomalies that may occur in the next time step based on current and past data.
                                                       - Providing the sensor values associated with the next time step.
                                                       - Assisting users in understanding the implications of predicted anomalies.
                                                       - Offering guidance on how to adjust the manufacturing process to avoid or mitigate predicted anomalies.
                                                       - Ensuring that users have the most accurate and up-to-date sensor readings for decision-making.""",

            "Production Forecasting": """You assist users with tasks related to predicting product demand in vegemite production pipeline. Your responsibilities include:
                                                               - Predicting next hour yeast product yield. """,
            "Causal analysis": """You assis users with causal discovery, causal analysis and root cause analysis"""

        }

        return system_templates

    @staticmethod
    def read_data():
        with open('/Users/chathurangishyalika/Custom_Compact_Copilot/SmartPilot/Agent 3: InfoGuide/src/assets/filtered_manufacturing_text.txt') as f:
            f_lines = f.read().splitlines()
            f_str = ''.join([re.sub(r'[^A-Za-z0-9 ]+', '' ,line) for line in f_lines if re.sub(r'[^A-Za-z0-9 ]+', '' ,line)])
        return f_str
