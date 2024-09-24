# Reliability and Maintenance Analysis Tool

### Overview
A tool in Python that can evaluate any machine failure dataset that automatically produces the relevant managerial information regarding that machine’s optimal maintenance policy.


The Reliability and Maintenance Analysis Tool (RMAT) is designed to assist in maintenance decision-making for various machines based on historical reliability and condition data. It allows users to analyze maintenance histories and condition-based data to determine optimal maintenance strategies, aiming to minimize maintenance costs while improving machine reliability.

RMAT is built in Python and supports Kaplan-Meier and Weibull reliability estimations, as well as cost analyses for both age-based and condition-based maintenance policies. The tool is scalable and can be applied to future machines or datasets following the same structure.
Features

For each machine in the dataset, RMAT can:

- Kaplan-Meier Estimates: Provide Kaplan-Meier estimates of the reliability function.
- Weibull Distribution: Estimate Weibull reliability functions based on Maximum Likelihood Estimates (MLE) of the two Weibull parameters.
- Combined Visualization: Generate a figure showcasing both the Kaplan-Meier and Weibull reliability estimates.
- Mean Time Between Failures (MTBF): Calculate the MTBF for both Kaplan-Meier and Weibull estimates.
- Age-Based Maintenance Policy:
  - Calculate the mean cost per unit time for age-based maintenance policies as a function of maintenance age TT.
  - Plot the maintenance cost as a function of TT.
- Condition-Based Maintenance (for machines with condition data):
  - Simulate and calculate the mean cost per unit time of a condition-based maintenance policy for different threshold values.
  - Provide a visualization of maintenance costs for varying thresholds.

### Getting Started
##### Prerequisites

Before running RMAT, ensure you have the following Python packages installed:
- numpy
- pandas
- matplotlib
- scipy
- lifelines

You can install these packages using the following command:

bash

pip install numpy pandas matplotlib scipy lifelines

### Input Data

RMAT works with datasets containing the following information for each machine:

- Costs for preventive and corrective maintenance for specific machines. 
- Historical maintenance data indicating the times of previous maintenance interventions.
- The dataset should also specify whether each intervention was preventive or corrective.
- For the third machine (and any others with condition data), the dataset should also include condition-based data, which is used for exploring condition-based maintenance policies.

Ensure that your input dataset follows the structure outlined in the manual provided. This ensures compatibility with the tool.
Running the Tool

    Analyze Kaplan-Meier and Weibull Estimates: The tool will estimate the reliability function using both Kaplan-Meier and Weibull methods. It will generate a plot showcasing these estimates and calculate the MTBF for both.

    Cost Analysis for Age-Based Maintenance: RMAT will calculate and plot the mean cost per unit time for an age-based maintenance strategy, varying the maintenance age TT.

    Condition-Based Maintenance Analysis: For machines with condition data, the tool simulates and calculates the cost per unit time for different condition-based maintenance thresholds. The tool also provides a plot showing these costs for different threshold values.

Example Usage

Once the data is prepared, you can run RMAT as follows:

python

# Import the tool
from rmat import RMAT

# Load your dataset
data = pd.read_csv("machine_data.csv")

# Create RMAT object and run analysis
rmat = RMAT(data)
rmat.run_analysis()

This will generate the required reliability estimates, MTBF, and cost function plots.
Future Extensibility

RMAT is built to be scalable and reusable. You can easily analyze new machines by providing datasets that follow the required structure. The tool can handle machines with or without condition data.

### Example screenshots
![Machine-1-cost](https://github.com/user-attachments/assets/90051582-70fa-4e5a-b8da-f3d1de5c7260)
![Machine-1-Reliability](https://github.com/user-attachments/assets/63719721-6678-4c43-a0e8-eb31d3a6071a)

### Contributions

Krzysztof Piotrowski

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Contact

For any questions or issues, please contact krzysztof.piotrowski.in@gmail.com

