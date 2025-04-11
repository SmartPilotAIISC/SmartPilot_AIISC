# 🤖SmartPilot:Agent-Based CoPilot for Intelligent Manufacturing

We introduce SmartPilot: 𝘼 𝘾𝙪𝙨𝙩𝙤𝙢, 𝘾𝙤𝙢𝙥𝙖𝙘𝙩 𝙖𝙣𝙙 𝙉𝙚𝙪𝙧𝙤𝙨𝙮𝙢𝙗𝙤𝙡𝙞𝙘 𝘼𝙄 𝙢𝙤𝙙𝙚𝙡 - The co-pilot that leverages a **custom**, **right_sized**, **neurosymbolic** AI model to transform manufacturing processes.

Visit here for the demo: [Demo link](https://youtu.be/etgkhH_8jS4)


# To Run SmartPilot: Follow these steps

Install dependencies:
```
pip install -r requirements.txt
```

To run the chatbot with Front-End

```
cd Agent 3: InfoGuide/src

streamlit run agent_run.py
```

## Core System Framework: Key Components and Technical Features of the SmartPilot Platform

![Core System Framework](https://github.com/ChathurangiShyalika/Manufacturing_Copilot/blob/master/SmartPilot.png "Core System Framework of SmartPilot")

## Method and Applications Details

### 🔍 Key Components for The SmartPilot
#### 1) Agent-based System: 
Consists of three customized agents.<br>
**i) PredictX**: An anomaly prediction agent identifies and predicts anomalies before they occur, alerting manufacturing teams in real-time to prevent disruptions.
<br>
**ii) ForeSight**: A demand forecasting agent analyzes product data to anticipate demand fluctuations, providing insights and alerts on unexpected events to ensure smooth operations.
<br>
**iii) InfoGuide**: An agent that acts as a Question-and-Answer chatbot, ready to assist with domain-specific queries and generate responses tailored to user needs.

#### 2) Multimodal Data:
It processes multimodal data, including time series sensor data and images for anomaly prediction, time series sensor data for forecasting, and manufacturing manuals (in text format) alongside time series sensor data for information retrieval.
It is trained on diverse manufacturing related datasets. 
The figure below illustrates the interactions between the three agents, showcasing their inputs, outputs, and the interconnections between the various components.
![Multimodal Integration Architecture](https://github.com/ChathurangiShyalika/Manufacturing_Copilot/blob/master/MultimodalArchitecture.png "Multimodal Architecture of SmartPilot")


#### 3) Custom, Compact and NeuroSymbolic model:
🔧 𝘾𝙪𝙨𝙩𝙤𝙢: Tailored to solve specific industry challenges (here focused on anomalies in assembly processes and demand in production processes), providing focused and practical solutions.
The system is designed to be highly flexible and customizable to selected industrial-based applications.

⚙️ 𝘾𝙤𝙢𝙥𝙖𝙘𝙩: Lightweight and cost-effective, optimized for real-time deployment on **edge devices**.
Each individual agent is small, operates efficiently using minimal computational overhead.

🧠 𝙉𝙚𝙪𝙧𝙤𝙨𝙮𝙢𝙗𝙤𝙡𝙞𝙘: Integrates curated data, manufacturing knowledge, and human expertise (subject matters) for enhanced reliability and safety.
We use manufacturing based process-ontologies, knowledge graphs and structured knowledge sources as knowledge sources.

#### 4) Real-time Deployment:
SmartPilot and its agents are deployable on edge-devices. SmartPilot and its agents are deployable on edge devices. It is currently deployed in two manufacturing facilities, one focused on toy rocket assembly and the other on Vegemite production process.

#### 5) Enterprise Architecture:
Responsible in integrating and scaling neurosymbolic models within manufacturing systems, ensuring seamless communication, scalability, and compliance across enterprise platforms like MES, ERP, and SCADA. EA also aligns the model with business goals, optimizes resource management, and enforces security protocols for real-time edge computing and data governance.


### ⚙️ Technical Features for The SmartPilot

| **Feature**                             | **Feature Description**                                                                                                                                                                                                                           | **Example and How we achieved?**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Alignment**                           | Ensuring that AI systems are designed and operate consistently with the enterprise's goals, values, and ethics, while also providing metrics for evaluating alignment with objectives.                                                            | SmartPilot ensure the model's actions and decisions support the company's objectives. Metrics such as reduced downtime, fewer production losses, and improved machine efficiency ensure the PredictX agent’s actions align with business objectives.                                                                                                                                                                                                                                                                |
| **Grounding**                           | Refers to connecting AI outputs to real-world concepts, data, and actions to ensure relevance and accuracy, which includes training AI on domain-specific data.                                                                                   | SmartPilot leverages diverse data sources, including multimodal data such as sensor readings and images, along with structured knowledge from knowledge graphs and process ontologies. It utilizes advanced deep learning models, including fusion models based on autoencoders, CNNs, and LSTMs, to process this information and support robust, context-aware decision-making.                                                                                                                                    |
| **Instructability**                     | The ability of an AI system to accept instructions from users and adapt its behavior, often by allowing symbolic input to override statistical models when needed.                                                                                | Allows for domain experts in manufacturing to influence AI outputs by integrating human expertise for handling anomalies in assembly processes. The SmartPilot system incorporates expert knowledge and provides mechanisms for human operators to adjust AI behavior when encountering anomalies in the assembly process. This enables instructability by letting users guide the AI, ensuring decisions remain aligned with operational realities and domain-specific nuances in manufacturing.                   |
| **Explainability and Interpretability** | The capability of an AI system to provide understandable reasons for its decisions or predictions, ensuring transparency for both technical and non-technical users.                                                                              | Provides reasoning and explanations behind the predictions, including reasoning behind predictions.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Safety**                              | Ensuring AI systems operate without causing unintended harm, including implementing fail-safe mechanisms and rigorous testing before deployment.                                                                                                  | Ensures that responses are safe and reliable, following expert guidelines. SmartPilot emphasizes safety by adhering to carefully curated expert knowledge, ensuring that the predictions are safe, effective, and in line with standards. InfoGuide agent prevents answering unrelated queries effectively eliminating hallucination.                                                                                                                                                                               |
| **Abstraction**                         | The process of simplifying complex real-world data and situations to allow AI systems to focus on relevant features for decision-making.                                                                                                          | Handles complex scenarios such as rare manufacturing events by abstracting key data from historical and real-time sources for actionable insights. SmartPilot utilizes abstraction by synthesizing real-time data and historical information, simplifying complex manufacturing scenarios (e.g., anomalies) into actionable insights. This allows the system to focus on the relevant features of a process, offering a streamlined approach to problem-solving in complex environments like manufacturing.         |
| **Embodiement**                         | Enables AI systems to intelligently interact with its real-time environment through the integration of sensory input, feedback, and adaptive behavior, allowing for dynamic responses and AI-enabled autonomy in complex industrial applications. | SmartPilot enables AI to model essential aspects such as cyber-physical interaction, AI-driven autonomy, and context awareness, allowing systems to operate intelligently in real-time environments. The integration of sensory inputs and feedback, along with adaptive behavior, enhances the system's capability to respond dynamically to changing conditions.                                                                                                                                                  |
| **Custom, Neurosymbolic, and Compact**  | Describes methods focused on creating AI systems that are tailored to specific industry needs, use neurosymbolic reasoning for explainability, and are optimized for lightweight deployment.                                                      | SmartPilot is built as custom solutions tailored to manufacturing industry, using neurosymbolic reasoning for explainability, and optimized for lightweight deployment.  The copilot is custom-built to address specific industry needs, using neurosymbolic AI for enhanced reasoning and explanation. The system is compact, meaning it is lightweight and optimized for real-time use on consumer-grade hardware, aligning with the principles of being cost-effective and accessible for real-world applications. |




