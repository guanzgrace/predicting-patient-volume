# Predicting Sick Patient Volume in a Pediatric Outpatient Setting using Time Series Analysis

Dynamic, one month ahead (oma), and one step ahead (osa) models are found in their respectively named folders.

To find the model architecture for our NN and RNN model, we sought to find which model had the best validation MSE on all visit data (Find_Best_Dynamic_NN.ipynb and find_best_rnn.py). Then, we ran each constant, baseline, and time series model once, and each neural network model five times to report the MSEs in Table 2. Lastly, we printed the predictions one trial of each model for graphing.