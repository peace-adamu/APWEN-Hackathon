# 🛠️ Milling Machine Failure Type Prediction App

## Abstract
This project presents a predictive maintenance solution for milling machines by applying machine learning techniques and deploying the final model using a user-friendly Streamlit web application. The dataset used includes critical machine parameters such as air temperature, process temperature, rotational speed, torque, and tool wear. The goal was to predict the type of failure the machine may encounter during operation.
To prepare the data, preprocessing steps such as label encoding, feature scaling, and handling class imbalance with SMOTE were applied. A Random Forest Classifier was trained using the resampled data and optimized through GridSearchCV. The final model demonstrated perfect predictive performance on the test set, achieving:
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1-score: 1.00
The confusion matrix confirmed flawless classification across all five failure types. This indicates the model's exceptional ability to generalize and correctly predict equipment failure conditions, enabling preemptive intervention.
The model was exported using joblib and integrated into a Streamlit interface, allowing end-users to input real-time machine metrics and receive instant predictions on possible failure modes. The solution empowers manufacturing teams to make data-driven maintenance decisions, significantly reducing downtime and increasing operational efficiency.

## Introduction
The study on the prediction of milling machine failure using machine learning focuses on developing a robust classification model capable of detecting potential failures in milling machines based on key operational parameters. This initiative is driven by the urgent need to improve machine uptime, reduce unplanned maintenance, and enhance productivity in manufacturing environments through the adoption of smart and proactive maintenance strategies.
Historically, maintenance practices evolved from reactive to preventive, but the limitations of these traditional methods — such as premature component replacements or costly downtimes — have made them inadequate for modern, complex machinery. As industries progress into the era of Industry 4.0, maintenance is no longer an afterthought but a strategic function, demanding intelligent systems that integrate data analytics, real-time monitoring, and machine learning to enable condition-based decision-making.
Predictive Maintenance (PM) has emerged as a transformative approach in this context. It involves anticipating machine failures before they occur, allowing for timely intervention that minimizes unnecessary maintenance and avoids catastrophic equipment breakdowns. A key component of PM is Condition Monitoring (CM), which tracks physical indicators such as temperature, torque, rotational speed, and tool wear to assess equipment health. In milling machines specifically, this monitoring forms part of Tool Condition Monitoring (TCM) — a vital strategy that directly affects surface quality, precision, and the overall economics of machining processes.
With advancements in the Internet of Things (IoT) and widespread deployment of industrial sensors, data acquisition is no longer a challenge. The focus has shifted to building intelligent models that can analyze this rich stream of sensor data to make accurate predictions. In this regard, Machine Learning (ML) techniques provide a powerful solution, capable of modeling complex, non-linear relationships and adapting to dynamic production conditions.
This project leverages a real-world dataset of milling machine operations, composed of five critical features: air temperature, process temperature, rotational speed, torque, and tool wear. After thorough preprocessing — including feature scaling, label encoding, and balancing the data using SMOTE — a Random Forest Classifier was trained and optimized using GridSearchCV to enhance performance.
The model achieved perfect classification metrics, with an accuracy, precision, recall, and F1-score of 1.00 across all failure categories. The outstanding performance confirms the effectiveness of Random Forest in capturing the subtle signals in the machine's operating conditions that indicate failure modes.
The trained model was then deployed using Streamlit, a lightweight Python web framework, to provide a real-time interface where users can input current machine conditions and receive immediate predictions on failure types. This integration of machine learning with interactive UI tools not only enhances usability but also empowers machine operators and maintenance engineers with actionable insights for better decision-making.
In conclusion, this study demonstrates the effectiveness of combining industrial sensor data with machine learning models for predictive maintenance. It highlights the potential of such systems to reduce downtime, extend equipment lifespan, and optimize overall operational efficiency in modern manufacturing setups.

## Description of the Random Forest Model
Random Forest is a robust supervised machine learning algorithm that operates by constructing an ensemble of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. This ensemble approach ensures better generalization, reduces overfitting, and enhances the predictive accuracy of the model.

Each individual tree in the forest is built from a bootstrap sample of the training data, a method that involves sampling with replacement. Additionally, when splitting a node during tree construction, the best split is found among a random subset of features, which introduces further diversity among the trees and contributes to the overall strength of the model.

In this study, the Random Forest Classifier was trained to predict failure types in milling machines using five critical input features: air temperature, process temperature, rotational speed, torque, and tool wear. The model aggregates the predictions of multiple decision trees and makes the final prediction based on majority voting. This ability to reduce variance while maintaining low bias makes Random Forest a preferred choice for classification tasks involving complex industrial datasets.

Classification Rule: C B
​
 (x)=majority_vote{Cb(x)} b=1B
​Where 
𝐶𝐵(𝑥)C B
​
 (x) is the final class prediction of the Random Forest, and 
𝐶
𝑏
(
𝑥
)
C 
b
​
 (x) represents the prediction of the 
𝑏
𝑡
ℎ
b 
th
  decision tree in the ensemble.

### Evaluation Criteria
To assess the performance and reliability of the Random Forest model, a set of standard evaluation metrics was employed. These metrics provide insight into both the predictive power and the classification quality of the model.

### Confusion Matrix
The confusion matrix is a tabular representation that compares the actual labels to those predicted by the model. For a multi-class classification task, it provides an intuitive summary of the prediction results through the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

- True Positives (TP): Correctly predicted positive class instances.
- True Negatives (TN): Correctly predicted negative class instances.
- False Positives (FP): Incorrectly predicted as positive (Type I error).
- False Negatives (FN): Incorrectly predicted as negative (Type II error).
This matrix enables a deeper understanding of the types of errors made by the model, especially in imbalanced datasets.

### Accuracy
Accuracy measures the proportion of correctly predicted instances among the total predictions made. It is a simple and effective metric in balanced classification tasks.

Accuracy
=
𝑇
𝑃
+
𝑇
𝑁
𝑇
𝑃
+
𝑇
𝑁
+
𝐹
𝑃
+
𝐹
𝑁
Accuracy= 
TP+TN+FP+FN
TP+TN
​ 
In this study, the tuned Random Forest model achieved an exceptional accuracy of 100%, demonstrating its effectiveness in learning the patterns related to machine failure from the input parameters.

### Precision, Recall, and F1 Score
To provide a more comprehensive evaluation, additional metrics such as precision, recall, and the F1 score were computed:
- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of true positive predictions among all actual positives.
- F1 Score: The harmonic mean of precision and recall, offering a balance between the two.
These metrics are especially useful in assessing model performance across multiple classes, as they provide class-level insights into misclassification tendencies and help in understanding model robustness.

