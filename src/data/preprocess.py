# preprocess.py - mon_standard.pkl, unmon_standard.pkl 전처리
import os
import pickle
import numpy as np
import pandas as pd
from itertools import groupby

# Constants
URL_PER_SITE = 10
TOTAL_MON_URLS = 950
TOTAL_UNMON_URLS = 10000
MON_LABEL = 1   # Label for monitored data
UNMON_LABEL = -1 # Label for unmonitored data

# -- Load Files --
def load_data():

    # load monitored dataset
    X1_mon, X2_mon, y_mon_cw, y_mon_ow = [], [], [], []
    with open('./data/pkl/mon_standard.pkl', 'rb') as mon_file: # Path to mon_standard.pkl
        mon_data = pickle.load(mon_file)
    for i in range(TOTAL_MON_URLS):
        for sample in mon_data[i]:
            size_seq, time_seq = [], []
            for c in sample:
                dr = 1 if c > 0 else -1
                time_seq.append(abs(c))
                size_seq.append(dr * 512)
            X1_mon.append(time_seq)
            X2_mon.append(size_seq)
            y_mon_cw.append(i//URL_PER_SITE) # Label for monitored data in closed world (95 websites)
            y_mon_ow.append(MON_LABEL) # Label for monitored data in open world (monitored(1))

    # load unmonitored dataset
    X1_unmon, X2_unmon, y_unmon = [], [], []
    with open('./data/pkl/unmon_standard10.pkl', 'rb') as unmon_file: # Path to unmon_standard10.pkl
        unmon_data = pickle.load(unmon_file)
    for i in range(TOTAL_UNMON_URLS):
        size_seq, time_seq = [], []
        for c in unmon_data[i]:
            dr = 1 if c > 0 else -1
            time_seq.append(abs(c))
            size_seq.append(dr * 512)
        X1_unmon.append(time_seq)
        X2_unmon.append(size_seq)
        y_unmon.append(UNMON_LABEL) # Label for unmonitored data in open world (unmonitored(-1))
    
    # open world - 2 classes(monitored(1) / unmonitored(-1))
    X1_total = X1_mon + X1_unmon
    X2_total = X2_mon + X2_unmon
    y_total_ow = y_mon_ow + y_unmon
    
    # open world - 96 classes(95 monitored + 1 unmonitored)
    y_total_combined = y_mon_cw + y_unmon

    # closed world
    df_cw = pd.DataFrame({
        'timestamps': X1_mon,
        'direction_size': X2_mon,
        'Label': y_mon_cw
    })
    
    # open world binary
    df_ow_binary = pd.DataFrame({
        'timestamps': X1_total,
        'direction_size': X2_total,
        'Label': y_total_ow
    })

    # open world multi
    df_ow_multi = pd.DataFrame({
        'timestamps': X1_total,
        'direction_size': X2_total,
        'Label': y_total_combined
    })

    return df_cw, df_ow_binary, df_ow_multi

# -- Utility Functions --
def calculate_cumulative_sum(data):
    current_sum = 0
    return [current_sum := current_sum + value for value in data]

def sum_same_direction(numbers):
    return [sum(group) for _, group in groupby(numbers, key=lambda x: x >= 0)]

# -- Main Feature Engineering --

def preprocess_to_csv(df, scenario):
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
    df['sum_incoming_outgoing_total_cum_diff'] = df['sum_incoming_outgoing_total'] - df['sum_incoming_outgoing_total_cum']
    df['sum_incoming_outgoing_total_burst_diff'] = df['sum_incoming_outgoing_total'] - df['sum_incoming_outgoing_total_burst']

    # -- Fraction -- 
    # Fraction of Incoming Packets, Outgoing Packets / Total Packets for each list
    # cumulative sum list has a samll outgoing portion(nearly 0) -> not needed.
    df['fraction_incoming'] = df['num_incoming'] / df['num_total']
    df['fraction_outgoing'] = df['num_outgoing'] / df['num_total']

    df['fraction_incoming_burst'] = df['num_incoming_burst'] / df['num_total_burst']
    df['fraction_incoming_burst_incoming_diff'] = df['fraction_incoming_burst']-df['fraction_incoming']
    df['fraction_outgoing_burst'] = df['num_outgoing_burst'] / df['num_total_burst']
    df['fraction_outgoing_burst_outgoing_diff'] = df['fraction_outgoing_burst']-df['fraction_outgoing']

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

    # Define file paths dynamically using scenario
    csv_filename = f'./data/csv/{scenario}_data.csv'
    cum_pickle_filename = f'./data/pkl/{scenario}_cumulative_data.pkl'
    burst_pickle_filename = f'./data/pkl/{scenario}_burst_data.pkl'

    # Save overall preprocessed data to CSV
    df_overall.to_csv(csv_filename, index=False)

    # Save cumulative and burst data to pickle files
    df_mon_cum = df[['cumulative_sum', 'Label']]
    df_mon_cum.to_pickle(cum_pickle_filename)

    df_mon_burst = df[['burst_sum', 'Label']]
    df_mon_burst.to_pickle(burst_pickle_filename)
    print(f"Data saved successfully for scenario: {scenario}")

def preprocess_data():
    df_cw, df_ow_binary, df_ow_multi = load_data()
    preprocess_to_csv(df_cw, scenario="closedworld")
    preprocess_to_csv(df_ow_binary, scenario="openworld_binary")
    preprocess_to_csv(df_ow_multi, scenario="openworld_multi")

def check_and_preprocess_data(closedworld_path,openworld_binary_path,openworld_multi_path):
    # 파일이 없는 경우
    if not (os.path.exists(closedworld_path) and os.path.exists(openworld_binary_path) and os.path.exists(openworld_multi_path)):
        print("Preprocessing data...")
        try:
            preprocess_data()  # 전처리 진행
            print("Preprocessing completed successfully.")
        except Exception as e:
            print(f"Error occurred during preprocessing: {e}")
    else:
        print("Preprocessed files exist.")