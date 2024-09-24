import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Using 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import math

data_path = ""  # Path to the directory where data files are located

#Prepares the dataset by adding a Duration column
def data_preparation(a):
    mf = pd.DataFrame(a) # Convert input data to a DataFrame

    #The Duration column is calculated as the difference between consecutive Time values.
    mf['Duration'] = mf['Time'].diff()

    #Handles missing values for the first failure
    mf.iloc[0, 2] = mf.iloc[0, 0]

    #The data is sorted to ensure that failures occur before censored events.
    mf.sort_values(by=['Duration', 'Event'], ascending=[True, False], inplace=True)
    prepared_data = mf.reset_index() # Resetting the index after sorting
    prepared_data = prepared_data.drop(columns='index') # Dropping the old index column
    return prepared_data # Returning the prepared DataFrame

#Calculates the Kaplan-Meier estimate of the reliability function.
def create_kaplanmeier_data(b):

    #Assigns an initial probability to each observation.
    b['Probability'] = 1 / len(b)

    for i in range(len(b)):

        #Checks if the Event for the current row is "PM" (preventative maintenance)
        event = b.loc[i, 'Event']
        if event == 'PM':

            #Calculates the probability of the next failure based on the remaining observations.
            p = b.loc[i, 'Probability']/len(b.loc[i+1:])

            #Updates the probabilities for the remaining observations by adding the calculated probability.
            b.loc[i + 1:, 'Probability'] = b.loc[i + 1:, 'Probability'].add(p)

            #Sets the probability for the current row to 0, as it is a censored observation.
            b.loc[i, 'Probability'] = 0
    for i in range(len(b)):

        #Checks if the Event for the current row is "failure".
        event = b.loc[i, 'Event']
        duration = b.loc[i, 'Duration']
        duration_below = b.loc[i + 1:, 'Duration'][b.loc[i + 1:, 'Event'] == 'failure']
        probability_below = b.loc[i + 1:, 'Probability'][b.loc[i + 1:, 'Event'] == 'failure']

        # If the event is a failure and there's a matching duration in the subsequent failures
        if event == 'failure' and (duration == duration_below).any():

            # Update the probability based on the matching durations
            b.loc[i, 'Probability'] += probability_below[duration_below == duration].sum()
            mask = (b.loc[i + 1:, 'Duration'] == duration) & (b.loc[i + 1:, 'Event'] == 'failure')
            b.loc[b.index[i + 1:][mask], 'Probability'] = 0  # Index the DataFrame directly with the mask

    #The reliability function is calculated as 1 minus the cumulative sum of probabilities.
    b['Reliability'] = 1 - b['Probability'].cumsum()

    #Ensure that the reliability values in the b DataFrame are never negative, and any extremely small values are rounded to 0.
    b.loc[b['Reliability']< 1e-10, 'Reliability'] = 0
    return b

#Calculates the MTBF using the Kaplan-Meier estimate.
def meantimebetweenfailures_KM(b):

    #Multiplies the Duration and Probability for each observation and sums the results.
    MTBF_KM = (b['Duration'] * b['Probability']).sum()
    return MTBF_KM

def fit_weibull_distribution(a):
    lambda_range = np.linspace(1, 45, 45)  # Range for the scale parameter (lambda)
    k_range = np.linspace(0.1, 4.5, 45)  # Range for the shape parameter (k)

    #create a data frame with each pair of parametres in each row
    com = pd.DataFrame([(l, k) for l in lambda_range for k in k_range], columns=['lambda', 'kappa'])

    for i in range(len(a)):
        durations = a.loc[i, 'Duration'] # Get the duration for the current observation
        if durations == 0:
            durations = a.loc[i, 'Duration'] + 1e-10 # Avoid division by zero
        for j in range(len(com)):
            l = com.loc[j, 'lambda'] # Current lambda value
            k = com.loc[j, 'kappa']  # Current kappa value
            if a.loc[i, 'Event'] == 'failure':

                # Calculate the Weibull probability density function (pdf) for failures
                fx = (k/l)*(durations/l)**(k-1)*math.exp(-(durations/l)**k)
                if fx == 0:
                    fx += 1e-10 # Avoid log(0) by adding a small number

                com.loc[j, 'Observation ' + str(i)] = np.log(fx) # Log-likelihood for failures
                com = com.copy()  # Ensure copy to avoid setting with copy warning
            else:

                # Calculate the Weibull reliability function for censored events
                Rx = math.exp(-(durations/l)**k)
                if Rx == 0:

                    # Avoid log(0) by adding a small number
                    Rx += 1e-10

                com.loc[j, 'Observation ' + str(i)] = np.log(Rx)
                com = com.copy()

    #Sum all columns in a row, containing 'Obervation'
    com['loglikelihood_sum'] = com.filter(like='Observation').sum(axis=1) # Total log-likelihood
    index_max = com['loglikelihood_sum'].idxmax()  # Index of maximum log-likelihood
    lam_value = com.loc[index_max, 'lambda']  # Best lambda value
    kap_value = com.loc[index_max, 'kappa']  # Best kappa value

    return lam_value, kap_value     # Returning the fitted Weibull parameters


def meantimebetweenfailures_weibull(l, k):

    # Calculate the Mean Time Between Failures (MTBF) for the Weibull distribution
    MTBF_weibull = l * math.gamma(1+1/k)
    return MTBF_weibull         # Returning the calculated MTBF


def create_weibull_curve_data(a, l, k):

    # Create a range of time values for Weibull reliability calculation
    if a['Duration'].max() < 100:
        t_range = np.linspace(0, int(a['Duration'].max()), int((a['Duration'].max()))*100)  #Fine granularity for short durations
    else:
        t_range = np.linspace(0, int(a['Duration'].max()), int((a['Duration'].max())) * 10) #Coarser granularity for long durations

    weibull_data = pd.DataFrame([t for t in t_range], columns=['t']) # Creating a DataFrame for time values
    weibull_data['R_t'] = np.exp(-(weibull_data['t'] / l) ** k)      # Calculating the Weibull reliability function
    return weibull_data

def visualisation(b, weibull_data, machine_name):

    # Creating a visualisation for the reliability functions
    fig1, ax1 = plt.subplots()   # Creating a figure and axis for plotting
    ax1.step(b['Duration'],  b['Reliability'], label='Kaplan-Meier estimate of the reliability function')    # Kaplan-Meier plot
    ax1.plot(weibull_data['t'],   weibull_data['R_t'], label='Weibull reliability function')     # Weibull plot
    ax1.set_xlabel('time (t)')       # Setting x-axis label
    ax1.set_ylabel('Reliability function R(t)')  # Setting y-axis label
    ax1.set_title(f'Goodness-of-fit for machine {machine_name}')  # Setting plot title
    ax1.legend()  # Displaying the legend
    plt.savefig(f'{data_path}Machine-{machine_name}-Reliability.png')  # Saving the plot as a PNG file


def create_cost_data(data, l, k, PM_cost, CM_cost, machine_name, mtbf):
    # Determine the maximum duration from the data
    max_duration = data['Duration'].max()
    # Set the interval for the range of durations, based on the maximum duration
    int_ = max_duration * 100 if max_duration < 100 else max_duration * 10

    # Create a DataFrame with a range of time values from 0 to the maximum duration
    dm = pd.DataFrame(np.linspace(0, max_duration, int(int_)), columns=['t'])
    # Calculate the reliability function R(t) using the Weibull parameters
    dm['R_t'] = np.exp(-(dm['t'] / l) ** k)
    # Calculate the failure function F(t) as the complement of R(t)
    dm['F_t'] = 1 - dm['R_t']
    # Calculate the probability density function f(t) for Weibull distribution
    dm['f_t'] = (k / l) * (dm['t'] / l) ** (k - 1) * np.exp(-(dm['t'] / l) ** k)

    sum = 0  # Initialize sum for calculating mean cycle length
    for i in range(1, len(dm)):
        # Calculate the width of the interval between consecutive time points
        width = dm['t'][i] - dm['t'][i - 1]
        # Calculate the midpoint of the current interval
        mid = (dm['t'][i] + dm['t'][i - 1]) / 2
        # Calculate the reliability at the midpoint
        R_t_mid = np.exp(-(mid / l) ** k)
        # Update the cumulative sum of reliability values
        sum += R_t_mid * width
        # Store the mean cycle length in the DataFrame
        dm.loc[i, 'mean_cycle_length'] = sum

    # Calculate the mean cost per cycle based on preventive and corrective maintenance costs
    dm['Mean_cost_per_cycle'] = CM_cost * dm['F_t'] + PM_cost * dm['R_t']
    # Calculate the cost rate as mean cost per cycle divided by mean cycle length
    dm['cost_rate'] = dm['Mean_cost_per_cycle'] / dm['mean_cycle_length']

    # Create a plot to visualize the cost rate over time
    fig2, ax2 = plt.subplots()  # Create a figure and axis for plotting
    x = dm['t']  # Time values for the x-axis

    # Limit cost rate values to a maximum of 200 for better visualization
    y = dm['cost_rate'].apply(lambda x: x if x < 200 else 200)
    # Plot the cost rate against time
    ax2.plot(x, y)
    ax2.set_xlabel('time (t)')  # Set x-axis label
    ax2.set_ylabel('Cost rate')  # Set y-axis label
    ax2.set_title(f'Cost rate for machine {machine_name}')  # Set plot title

    # Save the plot as a PNG file
    plt.savefig(f'{data_path}Machine-{machine_name}-cost.png')

    # Find the best cost rate and corresponding age for optimization
    best_cost_rate = dm['cost_rate'].min()  # Get the minimum cost rate
    best_age = dm.loc[dm['cost_rate'] == best_cost_rate, 't'].values[0]  # Get the age at which this cost rate occurs

    # Calculate the costs for corrective and preventive maintenance
    corrective_cost = 1 / mtbf * CM_cost
    preventive_cost = 1 / best_age * PM_cost

    # Calculate savings from the cost comparison
    savings = (corrective_cost - preventive_cost) / corrective_cost

    return best_age, best_cost_rate  # Return the best age and cost rate


def CBM_data_preparation(a, p, n):
    # Create a DataFrame from the input array 'a'
    df = pd.DataFrame(a)
    # Calculate the difference between consecutive conditions to find increments
    df['Increments'] = df['Condition'].diff()
    # Identify rows where increments are negative, indicating errors or inconsistencies
    remove_rows = df['Increments'] < 0
    # Prepare condition data by removing rows with negative increments
    prepared_condition_data = df.loc[~remove_rows]
    # Reset the index of the prepared DataFrame
    prepared_condition_data = prepared_condition_data.reset_index(drop=True)

    # Select rows from 'p' DataFrame where the event is 'failure'
    mac = p.loc[p['Event'] == 'failure']

    # Extract the failure level based on the times that match the failure events
    failures = prepared_condition_data.loc[prepared_condition_data['Time'].isin(mac['Time'].tolist()), 'Condition']
    failure_level = failures.iloc[0]

    # Save the prepared condition data to a CSV file
    prepared_condition_data.to_csv(f'{data_path}Prepared-Condition-Machine-{n}.csv')
    return failure_level, prepared_condition_data  # Return failure level and prepared data


def CBM_create_simulations(data, fl, range_):
    # Iterate over the specified range of thresholds
    for j in range_:
        time = 0  # Initialize time
        condition = 0  # Initialize condition
        simulations = np.empty((0, 2))  # Prepare an empty array to hold simulation results
        threshold = j * fl  # Calculate the threshold for preventive maintenance
        # Run 1000 simulations
        for i in range(1000):
            while True:  # Infinite loop until an event occurs
                time += 1  # Increment time
                # Randomly select a condition increment
                condition += np.random.choice(data['Increments'])
                if condition >= fl:  # Check for failure
                    event = 'failure'
                    row = np.array([time, event])  # Record the event
                    simulations = np.vstack([simulations, row])  # Append the event to simulations
                    break
                if condition > threshold:  # Check for preventive maintenance
                    event = 'PM'
                    row = np.array([time, event])  # Record the preventive maintenance event
                    simulations = np.vstack([simulations, row])  # Append the event to simulations
                    break
            # Create a DataFrame from the simulations results
            simulations_df = pd.DataFrame(simulations, columns=['Time', 'Event'])
            # Format the threshold for filename
            j_ = '{:.3f}'.format(j)
            # Save each simulation as a CSV file for better storage
            simulations_df.to_csv(f'{data_path}simulation_{j_}.csv')


def CBM_analyse_costs(data, range_, p, c, fl):
    # Call the function to create simulations for cost analysis
    CBM_create_simulations(data, fl, range_)
    dfs = []  # Initialize an empty list to hold DataFrames
    # Iterate over the range of thresholds
    for i in range_:
        # Format the threshold for filename
        i_ = '{:.3f}'.format(i)
        # Construct the filename to read the simulation results
        filename = f'{data_path}simulation_{i_}.csv'
        df = pd.read_csv(filename, index_col=0)  # Read the CSV file into a DataFrame
        # Rename columns to include the threshold
        df.columns = [f'{col}_{i_}' for col in df.columns]
        dfs.append(df)  # Append the DataFrame to the list
    # Concatenate all DataFrames along the columns
    combined_df = pd.concat(dfs, axis=1)
    # Filter to get columns related to time
    time_cols = combined_df.filter(regex='Time')
    time_avgs = time_cols.mean()  # Calculate average time for each threshold
    # Filter to get columns related to events
    event_cols = combined_df.filter(regex='Event')

    # Count occurrences of PM and failure events
    pm_counts = event_cols.apply(lambda col: col.value_counts().get('PM', 0), axis=0)
    failure_counts = event_cols.apply(lambda col: col.value_counts().get('failure', 0), axis=0)

    # Iterate over the range again to compute failure fractions
    for i in range_:
        i_ = '{:.3f}'.format(i)
        time_col = f'Time_{i_}'  # Time column for the current threshold
        event_col = f'Event_{i_}'  # Event column for the current threshold
        failures = event_cols[event_col].eq('failure').astype(int)  # Identify failures
        failure_time = failures * time_cols[time_col]  # Calculate time spent in failure
        failure_sum = failure_time.sum()  # Total failure time
        total_time = time_cols[time_col].sum()  # Total time
        failure_fraction = failure_sum / total_time  # Calculate failure fraction

    # Calculate mean cost per cycle based on PM and failure counts
    mean_cost_cycle = p * pm_counts / 1000 + c * failure_counts / 1000
    av = time_avgs.tolist()  # Convert average time to a list
    CBM_cost_rate = mean_cost_cycle / av  # Calculate cost rate
    return CBM_cost_rate  # Return the cost rate


def CBM_create_cost_data(data, p, c, fl, n):
    # Define the range of thresholds for running simulations
    range_ = np.arange(0.1, 1, 0.025)
    # Analyze costs based on the defined range
    CBM_cost_rate = CBM_analyse_costs(data, range_, p, c, fl)
    x = np.arange(0.1, 1, 0.025)  # Create an array of thresholds for plotting
    y = CBM_cost_rate.tolist()  # Convert cost rate to list
    fig3, ax3 = plt.subplots()  # Create a figure and axis for plotting
    ax3.plot(x, y)  # Plot cost rate against threshold
    ax3.set_xlabel('Maintenance threshold')  # Set x-axis label
    ax3.set_ylabel('Cost rate')  # Set y-axis label
    ax3.set_title(f'Cost rate per maintenance threshold for machine {n}')  # Set plot title
    min_idx = np.argmin(y)  # Find the index of the minimum cost rate
    CBM_threshold = x[min_idx]  # Get the threshold corresponding to the minimum cost rate
    plt.savefig(f'{data_path}Machine-{n}-condition-cost.png')  # Save the plot as a PNG file
    CBM_cost_rate = CBM_cost_rate.values[0]  # Extract the cost rate value
    return CBM_cost_rate, CBM_threshold  # Return the cost rate and the optimal threshold


def run_analysis():
    #data preparation
    machine_name = input('Which machine do you want to analyse?')
    machine_data = pd.read_csv(f"{data_path}Machine-{machine_name}.csv")
    prepared_data = data_preparation(machine_data)


    #Kaplan-Meier Estimation
    KM_Data = create_kaplanmeier_data(prepared_data)
    MTBF_KM = meantimebetweenfailures_KM(KM_Data)
    print('The MTBF Kaplan-Meier is:', MTBF_KM)

    #Weibull fitting
    lam_value, kap_value = fit_weibull_distribution(prepared_data)
    weibull_data = create_weibull_curve_data(prepared_data, lam_value, kap_value)
    MTBF_weibull = meantimebetweenfailures_weibull(lam_value, kap_value)
    print('The MTBF-Weibull is:', MTBF_weibull)

    #Visualisation
    visualisation(KM_Data, weibull_data, machine_name)

    # Input of costs
    cost_data = pd.read_csv(f'{data_path}Costs.csv')
    cd = pd.DataFrame(cost_data)
    cd_machine = cd.loc[cd['0'] == f'Machine {machine_name}']
    PM_cost = input('How much is the preventive maintenance cost? (press enter to use the given value): ')
    if PM_cost == '':
        PM_cost = cd_machine['PM cost'].iloc[0]
    else:
        PM_cost = float(PM_cost)
    CM_cost = input('How much is the corrective maintenance cost? (press enter to use the given value): ')
    if CM_cost== '':
        CM_cost = cd_machine['CM cost'].iloc[0]
    else:
        CM_cost = float(CM_cost)

    # age-based optimisation
    best_age, best_cost_rate = create_cost_data(prepared_data, lam_value, kap_value, PM_cost, CM_cost, machine_name, MTBF_weibull)
    print('The optimal maintenance age is:', best_age)
    print('The best cost rate:', best_cost_rate)

    while True:
        CBM_analysis = input('Is there condition information for this machine?')
        if CBM_analysis == 'no':
            break
        elif CBM_analysis == 'yes':
            condition_data = pd.read_csv(f'{data_path}Machine-{machine_name}-condition-data.csv')
            failure_level, prepared_condition_data = CBM_data_preparation(condition_data, machine_data, machine_name)

            CBM_cost_rate, CBM_threshold = CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name)
            print('The optimal cost rate under CBM is:', CBM_cost_rate)
            print('The optimal CBM threshold is:', CBM_threshold)
            break
    return
run_analysis()

