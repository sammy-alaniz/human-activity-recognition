### Summary of the Research Paper

The research paper titled "Predicting Blood Glucose with an LSTM and Bi-LSTM Based Deep Neural Network" by Qingnan Sun, Marko V. Jankovic, Lia Bally, and Stavroula G. Mougiakakou explores the use of a deep learning model combining Long Short-Term Memory (LSTM) and Bidirectional LSTM (Bi-LSTM) layers to predict future blood glucose (BG) levels in diabetes patients. The goal is to enable proactive management of hyperglycaemia and hypoglycaemia by forecasting BG levels over various prediction horizons (15, 30, 45, and 60 minutes). The proposed model outperforms baseline methods like ARIMA and SVR across all evaluation metrics, demonstrating its potential for real-time glucose monitoring applications, such as integration into an Artificial Pancreas system.

---

### Description of Graphs and Figures

1. **Figure 1: Structure of the LSTM Cell**
   - **Description**: This figure illustrates the internal architecture of an LSTM cell, highlighting its four key gates: input gate (i), forget gate (f), control gate (c), and output gate (o). It shows how these gates interact with the cell state (C) and hidden state (h) to process input data (x_t) and previous hidden state (h_{t-1}).
   - **Purpose**: To explain how LSTM addresses the vanishing gradient problem in traditional RNNs by selectively remembering or forgetting information over long time periods.

2. **Figure 2: General Structure of BRNN**
   - **Description**: This figure depicts the structure of a Bidirectional Recurrent Neural Network (BRNN), showing two separate sets of neurons processing the input sequence in forward and backward directions. The outputs from both directions are combined to produce the final result.
   - **Purpose**: To demonstrate how Bi-LSTM enhances prediction by incorporating context from both past and future time steps.

3. **Figure 3: Input and Output Dimensions of Each Layer**
   - **Description**: This figure outlines the architecture of the proposed prediction model, detailing the input and output dimensions of each layer: one LSTM layer (4 units), one Bi-LSTM layer (4 units), three fully connected layers (8, 64, and 8 units), and an output layer (1 unit for the predicted BG value).
   - **Purpose**: To provide a clear visual representation of the neural network’s structure and data flow.

---

### Review of the Research

#### Goals
The primary goal of the study was to develop and evaluate a deep learning model using LSTM and Bi-LSTM layers to predict future blood glucose levels based on continuous glucose monitoring (CGM) data. This predictive capability aims to assist diabetes patients, particularly those with Type 1 Diabetes (T1D), in preventing hyperglycaemic and hypoglycaemic events by providing timely warnings or enabling automated insulin delivery systems like the Artificial Pancreas. The study tested the model across multiple prediction horizons (15, 30, 45, and 60 minutes) and compared its performance against baseline methods (ARIMA and SVR).

#### Data Preprocessing
- **Handling Outliers and Missing Data**: 
  - Single outliers between two normal measurements were corrected using linear interpolation.
  - For periods with missing measurements, datasets were split into sub-datasets with valid continuous data. Sub-datasets with at least 1500 measurements (approximately 5 days) were selected for training and testing, while shorter segments were merged for pre-training.
- **Data Selection**: Only CGM measurements were used as features in this preliminary study, simplifying the model input to focus on glucose time series data.

#### Data Sources
- **Real Patient Data**: 
  - Obtained from a pilot study (NCT02546063, GoCARB) involving 20 T1D adults (mean age 35 ± 14 years, diabetes duration 17 ± 10 years). CGM data was collected every 5 minutes, yielding 6975 ± 1612 measurements per patient. After preprocessing, 26 sub-datasets (1791 ± 141 measurements) were used for training and testing, with shorter segments merged for pre-training.
- **In Silico Data**: 
  - Generated using the FDA-accepted UVa/Padova T1D Simulator, simulating 38-day trials for 11 virtual adult subjects. A virtual CGM with 5-minute sampling provided 120,395 glucose measurements, merged into a single file for pre-training.

#### Model Training
- **Architecture**: The model was built using Keras (version 2.0.8) in Python 3.4.3, consisting of one LSTM layer (4 units), one Bi-LSTM layer (4 units), three fully connected layers (8, 64, and 8 units), and a single-unit output layer.
- **Pre-Training**: 
  - Two rounds of pre-training were conducted to create a generalized "global model":
    1. First round used in silico data (120,395 samples).
    2. Second round used merged real patient data with fewer than 1500 measurements (93,443 samples).
  - Epoch numbers for pre-training were optimized experimentally (e.g., 1300 epochs for PH=30 minutes) to balance performance and computational cost.
- **Training and Testing**: 
  - The model was fine-tuned and tested on 26 real patient sub-datasets (67% training, 33% testing) with 100 epochs. Cross-validation was used to prevent overfitting.

#### Results
- **Evaluation Metrics**: Performance was assessed using Root Mean Square Error (RMSE), Correlation Coefficient (CC), Time Lag (TL), and Fit across four prediction horizons.
- **Comparative Performance**:
  - **PH = 15 minutes**: LSTM (RMSE: 11.633, CC: 0.974, TL: 9.423, Fit: 77.714) outperformed ARIMA (RMSE: 12.256, CC: 0.972, TL: 10.192, Fit: 76.425) and SVR (RMSE: 11.694, CC: 0.973, TL: 9.808, Fit: 77.565).
  - **PH = 30 minutes**: LSTM (RMSE: 21.747, CC: 0.909, TL: 20.385, Fit: 58.523) beat ARIMA (RMSE: 22.924, CC: 0.903, TL: 22.885, Fit: 55.923) and SVR (RMSE: 22.135, CC: 0.904, TL: 20.769, Fit: 57.644).
  - **PH = 45 minutes**: LSTM (RMSE: 30.215, CC: 0.818, TL: 32.692, Fit: 42.563) surpassed ARIMA (RMSE: 32.588, CC: 0.806, TL: 37.885, Fit: 37.463) and SVR (RMSE: 30.628, CC: 0.812, TL: 34.423, Fit: 41.595).
  - **PH = 60 minutes**: LSTM (RMSE: 36.918, CC: 0.722, TL: 46.346, Fit: 30.079) outperformed ARIMA (RMSE: 40.841, CC: 0.698, TL: 52.885, Fit: 21.694) and SVR (RMSE: 37.422, CC: 0.709, TL: 47.885, Fit: 28.893).
- **Key Findings**: The LSTM-based model consistently reduced RMSE and TL while increasing CC and Fit across all horizons, indicating superior predictive accuracy and timeliness. Optimal pre-training epochs (e.g., 1300 for PH=30) further enhanced performance.

#### Conclusion
The study successfully demonstrated that an LSTM and Bi-LSTM-based deep neural network can effectively predict blood glucose levels, outperforming traditional methods. Its reliance solely on CGM data makes it versatile for various diabetes management scenarios. Future work aims to incorporate additional features (e.g., insulin and meal data) and implement an alarm system for hypo- and hyperglycaemic events, enhancing its practical utility.