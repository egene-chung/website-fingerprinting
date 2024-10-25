import pickle
import numpy as np
import pandas as pd
from itertools import groupby

USE_SUBLABEL = False
URL_PER_SITE = 10
TOTAL_URLS   = 950

# Load mon_standard pickle file
print("Loading datafile...")
with open('./data/mon_standard.pkl', 'rb') as fi: # Path to mon_standard.pkl in Colab
    data = pickle.load(fi)

X1 = [] # Array to store instances (timestamps) - 19,000 instances, e.g., [[0.0, 0.5, 3.4, ...], [0.0, 4.5, ...], [0.0, 1.5, ...], ... [... ,45.8]]
X2 = [] # Array to store instances (direction*size) - size information
y = [] # Array to store the site of each instance - 19,000 instances, e.g., [0, 0, 0, 0, 0, 0, ..., 94, 94, 94, 94, 94]

# Differentiate instances and sites, and store them in the respective x and y arrays
# x array (direction*timestamp), y array (site label)
for i in range(TOTAL_URLS):
    if USE_SUBLABEL:
        label = i
    else:
        label = i // URL_PER_SITE # Calculate which site's URL the current URL being processed belongs to and set that value as the label. Thus, URLs fetched from the same site are labeled identically.
    for sample in data[i]:
        size_seq = []
        time_seq = []
        for c in sample:
            dr = 1 if c > 0 else -1
            time_seq.append(abs(c))
            size_seq.append(dr * 512)
        X1.append(time_seq)
        X2.append(size_seq)
        y.append(label)
size = len(y)

print(f'Total samples: {size}') # Output: 19000

# Create DataFrame
df = pd.DataFrame({
    'timestamps': X1,
    'direction_size': X2,
    'Label': y
})

# cumulative_sum
def calculate_cumulative_sum(data):
    cumulative_sum = []
    current_sum = 0
    for value in data:
        current_sum += value
        cumulative_sum.append(current_sum)
    return cumulative_sum

# burst_sum
def sum_same_direction(numbers):
    result = []
    for k, group in groupby(numbers, key=lambda x: x >= 0):
        result.append(sum(group))
    return result

# Calculate each cumulative/burst sum list
df['cumulative_sum'] = df['direction_size'].apply(calculate_cumulative_sum)
df['burst_sum'] = df['direction_size'].apply(sum_same_direction)

# -- Sum --
# Sum of Incoming Packets, Outgoing Packets, Total Packets per (normal, cumulative, burst)
df['num_incoming'] = df['direction_size'].apply(lambda x: sum(1 for i in x if i < 0))
df['num_outgoing'] = df['direction_size'].apply(lambda x: sum(1 for i in x if i > 0))
df['num_total'] = df['direction_size'].apply(len)  

df['num_incoming_cum'] = df['cumulative_sum'].apply(lambda x: sum(1 for i in x if i < 0))
df['num_outgoing_cum'] = df['cumulative_sum'].apply(lambda x: sum(1 for i in x if i > 0))
df['num_total_cum'] = df['cumulative_sum'].apply(len) 

df['num_incoming_burst'] = df['burst_sum'].apply(lambda x: sum(1 for i in x if i < 0))
df['num_outgoing_burst'] = df['burst_sum'].apply(lambda x: sum(1 for i in x if i > 0))
df['num_total_burst'] = df['burst_sum'].apply(len)

#  add counts of incoming, outgoing, and total number of packets of each list
df['sum_incoming_outgoing_total'] = df['num_incoming'] + df['num_outgoing'] + df['num_total']
df['sum_incoming_outgoing_total_cum'] = df['num_incoming_cum'] + df['num_outgoing_cum'] + df['num_total_cum']
df['sum_incoming_outgoing_total_burst'] = df['num_incoming_burst'] + df['num_outgoing_burst'] + df['num_total_burst']

# -- Fraction -- 
# Fraction of Incoming Packets, Outgoing Packets / Total Packets for each list
# cumulative sum list has a samll outgoing portion(nearly 0) -> not needed.
df['fraction_incoming'] = df['num_incoming'] / df['num_total']
df['fraction_outgoing'] = df['num_outgoing'] / df['num_total']

df['fraction_incoming_burst'] = df['num_incoming_burst'] / df['num_total_burst']
df['fraction_outgoing_burst'] = df['num_outgoing_burst'] / df['num_total_burst']

# --std--
# Std calculations for each list
# std_incoming / std_outcoming has the same size, which results in std 0
df['std_total'] = df['direction_size'].apply(lambda x: np.std(x, ddof=1))

df['std_incoming_cum'] = df['cumulative_sum'].apply(lambda x: np.std([i for i in x if i < 0], ddof=1) if len([i for i in x if i < 0]) > 1 else 0)
df['std_outgoing_cum'] = df['cumulative_sum'].apply(lambda x: np.std([i for i in x if i > 0], ddof=1) if len([i for i in x if i > 0]) > 1 else 0)
df['std_total_cum'] = df['cumulative_sum'].apply(lambda x: np.std(x, ddof=1))

df['std_incoming_burst'] = df['burst_sum'].apply(lambda x: np.std([i for i in x if i < 0], ddof=1) if len([i for i in x if i < 0]) > 1 else 0)
df['std_outgoing_burst'] = df['burst_sum'].apply(lambda x: np.std([i for i in x if i > 0], ddof=1) if len([i for i in x if i > 0]) > 1 else 0)
df['std_total_burst'] = df['burst_sum'].apply(lambda x: np.std(x, ddof=1))

# --7.average --
# Average calculations for each list
df['avg_incoming'] = df['direction_size'].apply(lambda x: np.mean([i for i in x if i < 0]) if any(i < 0 for i in x) else 0)
df['avg_outgoing'] = df['direction_size'].apply(lambda x: np.mean([i for i in x if i > 0]) if any(i > 0 for i in x) else 0)
df['avg_total'] = df['direction_size'].apply(np.mean)

df['avg_cum_incoming'] = df['cumulative_sum'].apply(lambda x: np.mean([i for i in x if i < 0]) if any(i < 0 for i in x) else 0)
df['avg_total_cum'] = df['cumulative_sum'].apply(np.mean)

df['avg_burst_incoming'] = df['burst_sum'].apply(lambda x: np.mean([i for i in x if i < 0]) if any(i < 0 for i in x) else 0)
df['avg_burst_outgoing'] = df['burst_sum'].apply(lambda x: np.mean([i for i in x if i > 0]) if any(i > 0 for i in x) else 0)
df['avg_total_burst'] = df['burst_sum'].apply(np.mean)


# --First 30 packets--
# Count incoming and outgoing packets within the first 30 entries of each list
df['num_incoming_thirty'] = df['direction_size'].apply(lambda x: sum(1 for i in x[:30] if i < 0))
df['num_outgoing_thirty'] = df['direction_size'].apply(lambda x: sum(1 for i in x[:30] if i > 0))
df['num_incoming_thirty_cum'] = df['cumulative_sum'].apply(lambda x: sum(1 for i in x[:30] if i < 0))
df['num_outgoing_thirty_cum'] = df['cumulative_sum'].apply(lambda x: sum(1 for i in x[:30] if i > 0))
df['num_incoming_thirty_burst'] = df['burst_sum'].apply(lambda x: sum(1 for i in x[:30] if i < 0))
df['num_outgoing_thirty_burst'] = df['burst_sum'].apply(lambda x: sum(1 for i in x[:30] if i > 0))

# --Last 30 packets--
# Count incoming and outgoing packets within the last 30 entries of each list
df['num_incoming_last_thirty'] = df['direction_size'].apply(lambda x: sum(1 for i in x[-30:] if i < 0))
df['num_outgoing_last_thirty'] = df['direction_size'].apply(lambda x: sum(1 for i in x[-30:] if i > 0))
df['num_incoming_last_thirty_cum'] = df['cumulative_sum'].apply(lambda x: sum(1 for i in x[-30:] if i < 0))
df['num_outgoing_last_thirty_cum'] = df['cumulative_sum'].apply(lambda x: sum(1 for i in x[-30:] if i > 0))
df['num_incoming_last_thirty_burst'] = df['burst_sum'].apply(lambda x: sum(1 for i in x[-30:] if i < 0))
df['num_outgoing_last_thirty_burst'] = df['burst_sum'].apply(lambda x: sum(1 for i in x[-30:] if i > 0))

# --timestamp--
df['avg_timestamps'] = df['timestamps'].apply(lambda x: sum(x) / len(x) if x else None)
df['max_timestamps'] = df['timestamps'].apply(lambda x: max(x) if x else None)
df['std_timestamps'] = df['timestamps'].apply(lambda x: np.std(x) if x else None)

# --timestamp intervals--
df['intervals'] = df['timestamps'].apply(lambda x: np.diff(x).tolist() if len(x) > 1 else [])

df['mean_interval'] = df['intervals'].apply(lambda x: np.mean(x) if x else None)
df['max_interval'] = df['intervals'].apply(lambda x: np.max(x) if x else None)
df['min_interval'] = df['intervals'].apply(lambda x: np.min(x) if x else None)
df['std_dev_interval'] = df['intervals'].apply(lambda x: np.std(x) if x else None)
df['range_interval'] = df['intervals'].apply(lambda x: np.ptp(x) if len(x) > 0 else None)

df_overall = df.drop(columns=['timestamps', 'direction_size', 'cumulative_sum', 'burst_sum', 'intervals'])
print(df_overall)

# Save overall preprocessed data to csv
csv_filename = './data/overall_data.csv'
df_overall.to_csv(csv_filename, index=False)

# Save cumulative data / burst data in pickle file
# ./mon_cumulative_data.pkl
df_mon_cum = df[['cumulative_sum', 'Label']]
df_mon_cum.to_pickle('./data/mon_cumulative_data.pkl')

# ./mon_burst_data.pkl
df_mon_burst = df[['burst_sum', 'Label']]
df_mon_burst.to_pickle('./data/mon_burst_data.pkl')