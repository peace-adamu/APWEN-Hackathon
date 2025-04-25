# 🛠️ Milling Machine Failure Type Detection App

## Link to Milling Machine Failure Type Detection App
<a href= "https://apwen-hackathon-gprxts7zcmdtosupl849gq.streamlit.app/" view>To access the app</a>

## Table of Content
- [Acknowledgment](#acknowledgment)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Materials and Method](#materials-and-method)
- [Results and DisCcussion](#results-and-discussion)
- [Process Simulation Integration Using Aspen HYSYS](#process-simulation-integration-using-aspen-hysys)
-  [Conclusion](#conclusion)
- [References](#references)
## Acknowledgment
This work was completed as part of a hackathon organized by the Association of Professional Women Engineers of Nigeria (APWEN) Lagos SHENoVATION. Special thanks to Olayinka Adewumi and all academic mentors and technical advisors for their valuable contributors in the course of this hackathon.

## Abstract
This project presents an end-to-end predictive maintenance system for milling machines, integrating machine learning with chemical process simulation to enhance industrial reliability and performance. Using a dataset comprising critical operational metrics—air temperature, process temperature, rotational speed, torque, and tool wear—a Random Forest Classifier was trained to accurately predict five distinct types of machine failure. Comprehensive data preprocessing was applied, including label encoding, feature scaling, and class imbalance correction using SMOTE. The optimized model, tuned with GridSearchCV, achieved perfect classification metrics on the test set:
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-score: 1.00

The model was deployed through an intuitive Streamlit web application, enabling real-time input of machine parameters and delivering instant predictions of potential failure modes. This empowers engineers and plant operators to make informed, data-driven maintenance decisions—minimizing unplanned downtime and improving equipment longevity.

To demonstrate the practical impact of predicted failures, the project incorporates a chemical process simulation using DWSIM, modeling a simplified thermal loop that mimics real-world manufacturing environments. Key components—pumps, control valves, and heat exchangers—were configured to respond to ML-predicted failures:

Tool wear affects valve precision and flow stability.
Excessive torque or heat reduces pump efficiency and introduces cavitation risk.
Combined failure conditions simulate cascading effects through the system.
By mapping machine learning outputs to simulation triggers in DWSIM, the system visualizes how mechanical anomalies propagate through chemical processes—creating pressure drops, temperature deviations, and system alarms. This hybrid approach bridges data science and process engineering, offering a tangible, scenario-based evaluation of predictive maintenance strategies aligned with Industry 4.0 goals.

Ultimately, this integrated solution provides both prediction and visualization, making it a powerful tool for proactive fault detection, operator training, and safer plant design.

## 1.0 Introduction
The study on the prediction of milling machine failure using machine learning focuses on developing a robust classification model capable of detecting potential failures in milling machines based on key operational parameters. This initiative is driven by the urgent need to improve machine uptime, reduce unplanned maintenance, and enhance productivity in manufacturing environments through the adoption of smart and proactive maintenance strategies.
Historically, maintenance practices evolved from reactive to preventive, but the limitations of these traditional methods — such as premature component replacements or costly downtimes — have made them inadequate for modern, complex machinery. As industries progress into the era of Industry 4.0, maintenance is no longer an afterthought but a strategic function, demanding intelligent systems that integrate data analytics, real-time monitoring, and machine learning to enable condition-based decision-making.
Predictive Maintenance (PM) has emerged as a transformative approach in this context. It involves anticipating machine failures before they occur, allowing for timely intervention that minimizes unnecessary maintenance and avoids catastrophic equipment breakdowns. A key component of PM is Condition Monitoring (CM), which tracks physical indicators such as temperature, torque, rotational speed, and tool wear to assess equipment health. In milling machines specifically, this monitoring forms part of Tool Condition Monitoring (TCM) — a vital strategy that directly affects surface quality, precision, and the overall economics of machining processes.
With advancements in the Internet of Things (IoT) and widespread deployment of industrial sensors, data acquisition is no longer a challenge. The focus has shifted to building intelligent models that can analyze this rich stream of sensor data to make accurate predictions. In this regard, Machine Learning (ML) techniques provide a powerful solution, capable of modeling complex, non-linear relationships and adapting to dynamic production conditions.
This project leverages a real-world dataset of milling machine operations, composed of five critical features: air temperature, process temperature, rotational speed, torque, and tool wear. After thorough preprocessing — including feature scaling, label encoding, and balancing the data using SMOTE — a Random Forest Classifier was trained and optimized using GridSearchCV to enhance performance.
The model achieved perfect classification metrics, with an accuracy, precision, recall, and F1-score of 1.00 across all failure categories. The outstanding performance confirms the effectiveness of Random Forest in capturing the subtle signals in the machine's operating conditions that indicate failure modes.
The trained model was then deployed using Streamlit, a lightweight Python web framework, to provide a real-time interface where users can input current machine conditions and receive immediate predictions on failure types. This integration of machine learning with interactive UI tools not only enhances usability but also empowers machine operators and maintenance engineers with actionable insights for better decision-making.
In conclusion, this study demonstrates the effectiveness of combining industrial sensor data with machine learning models for predictive maintenance. It highlights the potential of such systems to reduce downtime, extend equipment lifespan, and optimize overall operational efficiency in modern manufacturing setups.

## 2.0 Materials and Method
### 2.1 Data Collection
The supervised machine learning model was developed using milling machine data collected at the Project Development Institute (PRODA), Enugu State. The dataset consists of 10,000 observations with five input parameters: air temperature, process temperature, rotational speed, torque, and tool wear.
### 2.2 Model Building
The Random Forest Classifier model was developed using these inputs. The pipeline used included 100 estimators. The dataset was split into a 70:30 ratio for training and testing, respectively.

## 3.0 Methodology
### 3.1 Description of the Random Forest Model
Random Forest is a robust supervised machine learning algorithm that operates by constructing an ensemble of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. This ensemble approach ensures better generalization, reduces overfitting, and enhances the predictive accuracy of the model.

Each individual tree in the forest is built from a bootstrap sample of the training data, a method that involves sampling with replacement. Additionally, when splitting a node during tree construction, the best split is found among a random subset of features, which introduces further diversity among the trees and contributes to the overall strength of the model.

In this study, the Random Forest Classifier was trained to predict failure types in milling machines using five critical input features: air temperature, process temperature, rotational speed, torque, and tool wear. The model aggregates the predictions of multiple decision trees and makes the final prediction based on majority voting. This ability to reduce variance while maintaining low bias makes Random Forest a preferred choice for classification tasks involving complex industrial datasets.
Classification Rule: C<sub>B</sub>(x)=majority_vote{C<sub>b</sub>(x)}<sup>B</sup><sub>b</sub>=1
​Where 
𝐶<sub>𝐵</sub>(𝑥) is the final class prediction of the Random Forest, and 𝐶<sub>𝑏</sub>(𝑥) represents the prediction of the b<sup>th</sup> decision tree in the ensemble.


### 3.2 Evaluation Criteria
To assess the performance and reliability of the Random Forest model, a set of standard evaluation metrics was employed. These metrics provide insight into both the predictive power and the classification quality of the model.

### 3.3 Confusion Matrix
The confusion matrix is a tabular representation that compares the actual labels to those predicted by the model. For a multi-class classification task, it provides an intuitive summary of the prediction results through the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

- True Positives (TP): Correctly predicted positive class instances.
- True Negatives (TN): Correctly predicted negative class instances.
- False Positives (FP): Incorrectly predicted as positive (Type I error).
- False Negatives (FN): Incorrectly predicted as negative (Type II error).
This matrix enables a deeper understanding of the types of errors made by the model, especially in imbalanced datasets.

### 3.4 Accuracy
Accuracy measures the proportion of correctly predicted instances among the total predictions made. It is a simple and effective metric in balanced classification tasks.

Accuracy = $\frac{TP + TN}{TP + FP + TN + FN}$

In this study, the tuned Random Forest model achieved an exceptional accuracy of 100%, demonstrating its effectiveness in learning the patterns related to machine failure from the input parameters.

### 3.5 Precision, Recall, and F1 Score
To provide a more comprehensive evaluation, additional metrics such as precision, recall, and the F1 score were computed:
- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of true positive predictions among all actual positives.
- F1 Score: The harmonic mean of precision and recall, offering a balance between the two.
These metrics are especially useful in assessing model performance across multiple classes, as they provide class-level insights into misclassification tendencies and help in understanding model robustness.


## 4.0 Results and Discussion
### 4.1 Tuned Random Forest Model Performance
After hyperparameter optimization using GridSearchCV, the Random Forest model was fine-tuned with the
following parameters:
- max_depth: None
- min_samples_leaf: 1
- min_samples_split: 2
- n_estimators: 100
This configuration yielded perfect performance on the test dataset, as detailed in Table 4.2.
#### Table 4.1: Classification Report for Tuned Random Forest Model
Class Label | Precision | Recall | F1-Score | Support
------------|-----------|--------|----------|--------
0 | 1.00 | 1.00 | 1.00 | 395
1 | 1.00 | 1.00 | 1.00 | 410
2 | 1.00 | 1.00 | 1.00 | 415
3 | 1.00 | 1.00 | 1.00 | 393
4 | 1.00 | 1.00 | 1.00 | 387
![confusion_matrix](https://github.com/user-attachments/assets/3d7a8e4f-98c0-4689-bea0-83df144116e4)

![tuned_confusion_matrix](https://github.com/user-attachments/assets/f9543cb0-ba8e-4f26-80cc-37ce66cbcc70)

- Overall Accuracy: 1.00 (2000 samples)
These results reflect an exceptional level of generalization by the model across all failure classes. Every
class was perfectly identified, with no false positives or false negatives, indicating the model's robustness and
reliability for real-world predictive maintenance deployment.


## 5.0 Process Simulation Integration Using Aspen HYSYS

To explore the downstream implications of milling machine failures on broader engineering systems, Aspen HYSYS was integrated as a process simulation layer.  The primary goal of integrating Aspen HYSYS is to simulate the downstream chemical processing impact of potential mechanical failures identified by the machine learning model. This aims to demonstrate how predictive maintenance decisions on the milling machine can influence thermal process efficiency, energy consumption, or safety margins in a broader industrial plant.

This multidisciplinary linkage provides a predictive maintenance chain that spans from mechanical tooling operations to chemical process safety, ensuring end-to-end operational reliability.

### 5.1 System Setup in Simulation Process (DWSIM)
✅ 1. Process Overview
A simplified fluid heating system was simulated using DWSIM, demonstrating how mechanical failures (such as those due to faulty machined parts like valves, pumps, or seals) affect process efficiency. The system models normal and failure scenarios in a parallel loop setup.

✅ 2. Parameters from ML Model
The following failure indicators predicted by the ML model were linked to process simulation conditions:
- Rotational Speed
- Torque
- Air Temperature
- Process Temperature
- Tool Wear

These were mapped to relevant process variables in DWSIM:

#### ML Model Output	DWSIM Parameter	Simulated Impact
- Tool Wear: Valve opening percentage ↓	Flow rate instability, pressure fluctuation
- Heat Dissipation Issue:	Pump temperature ↑	Efficiency drop, cavitation risk
- Torque: Overload	Pump head ↑	Energy imbalance, pressure surges
- Combined Failures:	Multi-point deviations	System alarms, operational instability
  
### 5.2 Example Simulation Scenario
✅ Modeled Process: Thermal Loop for Heated Fluid Transfer

Main Components:
- Two Pumps (Normal and Failure loops)
- Two Valves (Control flow in each path)
- Heat Exchanger (Processes output from the normal loop)
- Input & Output Streams (for Normal and Failure paths)

### 5.3 Simulation Logic
Dynamic behavior based on ML-predicted failure types:

| **ML Failure Type**        | **DWSIM Trigger**         | **System Impact**                              |
|---------------------------|---------------------------|------------------------------------------------|
| Tool Wear Failure         | Valve opening ↓           | Reduced flow, erratic pressure                 |
| Heat Dissipation Failure  | Pump temperature ↑        | Reduced efficiency, pump instability           |
| Torque Overload           | Pump pressure ↑           | High power draw, possible cavitation           |
| Combined Failures         | Multiple triggers         | Tripped flow loop, fault alarms                |

### 5.4 Workflow Summary
1. ML Prediction: An ML model flags a potential mechanical failure.
2. Simulation Setup: DWSIM variables are updated based on prediction (e.g., valve partially closed, pump runs hotter).
3. System Behavior: Deviations in output stream pressure, temperature, or flow rate are observed.
4. Output Reports: Efficiency loss, safety limits breached, or unstable flows are reported visually.

### 5.5 Benefits of Integration
- 🎯 Engineering Insight: Shows how small mechanical faults lead to chemical inefficiencies.
- ⚙️ Scenario Testing: Easy simulation of “What-if” mechanical failure events.
- 🔔 Predictive Maintenance: DWSIM simulation driven by ML alerts supports better plant decisions.
- 🌐 Industry 4.0 Ready: Bridges machine learning with process engineering.

### 5.6 To Anticipate Failure Prediction First, Then Simulation 

🔹 Step-by-Step:
- ML app analyzes machine sensor data (like torque, speed, tool wear) and predicts a failure type.
- Based on the predicted failure, you go into DWSIM and simulate the process response:

E.g., if “Tool Wear” is predicted → You simulate reduced valve opening %.

If “Overstrain” is predicted → You increase pump head or decrease efficiency.

You observe the chemical process behavior during the simulated failure.

✅ Why this is better:
It mirrors real life — engineers use ML to anticipate failures, then simulate to test safety, plan maintenance, or redesign the system.

###### Note: Keeps the simulation dynamic, based on real-time or historical predictions.

#### 🔁 Option 2: Simulation First, Then ML Prediction
Less common, but used for model training or validation.

You simulate various failure conditions in DWSIM first, generate data, and then use that data to train or test your ML model. 


## 6.0 Conclusion
This study presented a robust machine learning-based approach for predictive maintenance in milling
operations. Leveraging sensor data inputs - air temperature, process temperature, rotational speed, torque,
Predictive Maintenance Using Tuned Random Forest Model
and tool wear - the Random Forest classifier was trained to predict various failure conditions with high
accuracy.
Upon tuning, the model achieved 100% precision, recall, and F1-score across all classes, including:
- No failure
- Tool wear failure
- Heat dissipation failure
- Over strain failure
- Combined failure types

The results affirm the model's potential for industrial application, allowing operators to preemptively identify
specific failure types with perfect confidence, thereby minimizing machine downtime, reducing maintenance
costs, and enhancing operational efficiency.
Future Work:
- Deployment of the model in a real-time monitoring system for live predictions.
- Integration with IoT platforms for automated sensor data acquisition.
- Comparison with other ensemble models like XGBoost and LightGBM.
- Exploration of deep learning models for even more complex pattern detection.
The outcomes of this research not only validate the efficacy of Random Forest for industrial failure prediction
but also establish a scalable blueprint for data-driven maintenance strategies in smart manufacturing
environments.


## References
- Bruno, G., & Lombardi, F. (2020). Machine Learning Application for Tool Wear Prediction in Milling. Politecnico di Torino, Corso di Laurea Magistrale in Automotive Engineering (Ingegneria Dell'Autoveicolo).

- Rao, V. N. (2020). Machine Learning Application for Tool Wear Prediction in Milling (Master’s thesis, Politecnico di Torino). Rel. Giulia Bruno, Franco Lombardi.

