## **Data Preprocessing of `mon_standard.pkl`**

This guide provides detailed steps on how to preprocess the data stored in `mon_standard.pkl` to generate `overall_data.csv`. 
It processes the data into different lists, calculates cumulative sums, burst sums, packet counts, averages, and other statistical features, and finally saves the results in a CSV file. Pickle files are not uploaded due to storage issues.

### 1. **Load Data**
The first step is to load the dataset `mon_standard.pkl` to the data folder. 

### 2. **Run Code**

```bash
python preprocess_mon.py
```

### 3. **Cumulative and Burst Data Saved in data folder**

Additionally, cumulative and burst data are saved as separate pickle files:

```python
df_mon_cum = df[['cumulative_sum', 'Label']]
df_mon_cum.to_pickle('./data/mon_cumulative_data.pkl')

df_mon_burst = df[['burst_sum', 'Label']]
df_mon_burst.to_pickle('./data/mon_burst_data.pkl')
```
