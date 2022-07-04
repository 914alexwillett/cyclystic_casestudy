import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime as dt
import time
# for calcuating minimum ride distance between two station's lat/longs
from math import radians, cos, sin, asin, sqrt

"""Load and Generate Data Functions"""
def combine_data(_master_df, date_index):
    raw_df = load_data(date_index)
    master_df = _master_df
    master_df = master_df.append(raw_df, ignore_index=True)
    return master_df

def load_data(date_index):
    path = r'data/{}-divvy-tripdata.csv'.format(date_index)
    df = pd.read_csv(path)
    return df

def generate_data(date_indices: list):
    master_df = load_data(date_indices[0])
    for date_index in date_indices[1:]:
        master_df = combine_data(master_df, date_index)
        
    master_df = master_df.sort_values(by=['started_at']).reset_index()
    master_df = master_df.drop(['index'], axis=1)
    return master_df


"""Helper Functions"""
def distance(lat1, lat2, lon1, lon2, units='kilometers'):
    
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    if units == 'kilometers':
        r = 6371
    elif units == 'miles':
        r = 3958.8
    elif units == 'meters':
        r = 6371 * 1000
        
    else:
        return error, 'Please specificy units (miles or kilometers)'

    # calculate the result
    return (c * r)

# function to return key for any value
def get_key(val, _dict):
    for key, value in _dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def plot_stuff(data_dict):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.grid(axis='x')
    for item in data_dict:
        label_str = item
        ys = data_dict[item]
        xs = range(len(ys))
        ax.plot(xs, ys, marker='o', label=label_str)
    plt.legend()
    plt.show()
    
def filter_unique_list(_list):
    unique_list = []
    for x in _list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list 

"""Null Values and Error Functions"""
def percent_null(_df, column_name: str, verbose=False):
    df = _df
    null_df = df.loc[pd.isna(df[column_name])]
    pcnt = round(len(null_df) / len(df) * 100, 3)
    if verbose:
        print(f'{pcnt}% null {column_name} values')
        print(f'{len(null_df)} null {column_name} out of {len(df)}')
    return pcnt

def nonevent_df_filter(df):
    # return all from from data frame where: 
    # 1: riders end at the same station they started with and
    # 2: total ride time under 2mins (1.2e11 nanoseconds)
    return df.loc[(df['start_station_id'] == df['end_station_id']) & (df['tr_time_nano'] < 1.2e11)]

def filter_negAndNone(_list: list) -> list:
    list_filter = filter(lambda item: item > 0, _list)
    return list(list_filter)

def hard_coded_outlier_removal(df):
    df = df.loc[df['distance_miles'] < 50.]
    return df

def clean_data(df):
    # filter nonevents
    df = df.loc[np.invert((df['start_station_id'] == df['end_station_id']) & (df['tr_time_nano'] < 1.2e11))]
    # negative time_traveled
    df = df.loc[df['tr_time_nano'] > 0]
    # nulls
    df = df.loc[df['distance_miles'].notnull()]
    # outlier (hardcoded)
    df = df.loc[df['distance_miles'] < 50]
    return df

"""Data Generating Functions"""
def generate_columns_pack(df):
    df['distance_miles'] = [distance(df['start_lat'][x], df['end_lat'][x], df['start_lng'][x], df['end_lng'][x], units='miles') for x in range(len(df))]
    df['travel_time'] = [travel_time(df['started_at'][x], df['ended_at'][x]) for x in range(len(df))]
    time_delta_to_nano(df)
    weekday_num(df)
    return df

def travel_time(_start_time, _end_time):
    start_time = _start_time
    start_time = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = _end_time
    end_time = dt.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    time_change = end_time - start_time
    return time_change

def weekday_num(df):
    # (Mon, 0) (Tue, 1) (Wed, 2) (Th, 3) (Fri, 4) (Sat, 5) (Sun, 6)
    df['weekday_num'] = [dt.datetime.strptime(df.iloc[x, df.columns.get_loc('started_at')], "%Y-%m-%d %H:%M:%S").weekday() for x in range(len(df))]
    return df

def nano_to_mins(_nanos):
    seconds, nanos = divmod(_nanos, 1e9)
    nanos = int(nanos)
    mins, seconds = divmod(seconds, 60)
    mins_decimal = seconds/60
    value = mins + mins_decimal
    return value

def nano_to_time_str(_nano, incl_nano=False):
    seconds, nanos = divmod(_nano, 1e9)
    nanos = str(int(nanos))
    mins, seconds = divmod(seconds, 60)
    seconds = str(int(seconds))
    hours, mins = divmod(mins, 60)
    mins = str(int(mins))
    days, hours = divmod(hours, 24)
    hours = str(int(hours))
    days = str(int(days))
    if incl_nano:
        time_str = f'{days} days {hours}:{mins.zfill(2).zfill(2)}:{seconds.zfill(2)}.{nanos.zfill(9)}'
    else:
        time_str = f'{days} days {hours}:{mins.zfill(2).zfill(2)}:{seconds.zfill(2)}'
    return time_str

def time_delta_to_nano(df):
    df['tr_time_nano'] = pd.to_numeric(df['travel_time'], downcast='integer')
    return df

def leisure_df_filter(df):
    # return all from from data frame where: 
    # 1: riders end at the same station they started with and
    # 2: total ride time is above 10 mins (6e11 nanoseconds)
    return df.loc[(df['start_station_id'] == df['end_station_id']) & (df['tr_time_nano'] > 6e11)]

def nonleisure_df_filter(df):
    return df.loc[df['start_station_id'] != df['end_station_id']]

def distance_percentiles_df(df, index_name, units='miles'):
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    distance_pcnts = []
    if units == 'miles':
        df = df[df['distance_miles'].notnull()]
        for percentile in percentiles:
            distance_pcnts.append(round(np.percentile(df['distance_miles'], percentile), 3))
    elif units == 'meters':
        df = df[df['distance_meter'].notnull()]
        for percentile in percentiles:
            distance_pcnts.append(round(np.percentile(df['distance_meters'], percentile), 3))
    
    percentiles_df = pd.DataFrame([distance_pcnts], columns = [percentiles], index = [index_name])
    return percentiles_df

def pcnt_of_df(df, total_df, name, member_casual, subgroup=None):
    total_count = len(total_df)
    pcnt_total = round(len(df) / total_count * 100, 2)
    pcnt_type = round(len(df) / len(total_df[total_df['member_casual'] == member_casual]) * 100, 2)
    pcnt_dict = {'count': len(df), 'out of total': [pcnt_total], 
                 f'out of {member_casual}': [pcnt_type]}
    
    if subgroup is not None:
        for item in subgroup:
            pcnt_dict[item] = round(len(df) / len(subgroup[item] * 100), 2) 
            
    pcnt_df = pd.DataFrame(pcnt_dict, index = [name])
    
    return pcnt_df

def pcnt_of_dict(count, total_df, name, member_casual, subgroup=None):
    total_count = len(total_df)
    pcnt_total = round(count / total_count * 100, 2)
    pcnt_type = round(count / len(total_df[total_df['member_casual'] == member_casual]) * 100, 2)
    pcnt_dict = { 'name': name,
                  'count': count, '/ OUT OF TOTAL': pcnt_total, 
                 f'/ OUT OF {member_casual.upper()}': pcnt_type}
    
    if subgroup is not None:
        for item in subgroup:
            item_str = f'/ {item.upper()}'
            pcnt_dict[item_str] = round(count / len(subgroup[item]) * 100, 2) 
    
    return pcnt_dict

def print_markdown_pcnt(dictionary, name, criteria):
    print(f'#### {name} By {criteria}')
    for item in dictionary:
        print_list = []
        for i in item:
            print_list.append((i, item[i]))
        print(f'__{print_list[1][1]} {print_list[0][1]}__')
    
        for i in print_list[2:]:
            print(f'{i[1]} % {i[0]}')

"""Stats based on days of the week"""
def generate_wkd_stats(df, plot=False):
    # (Mon, 0) (Tue, 1) (Wed, 2) (Th, 3) (Fri, 4) (Sat, 5) (Sun, 6)
    weekday_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Th': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    weekday_count = Counter(df['weekday_num']).most_common()
    weekday_dict = {}
    for item in weekday_count:
        day_str = get_key(item[0], weekday_map)
        count = item[1]
        month_pcnt = round(count / len(df) * 100, 2)
        weekday_dict[item[0]] = (day_str, count)

    if plot:
        ys = [weekday_dict[i][1] for i in range(0, 7)]
        xs = [weekday_dict[i][0] for i in range(0, 7)]

        plt.figure(figsize=(12, 8))
        plt.plot(xs, ys)
        plt.show()
        
    return weekday_dict

def gen_wkd_stat(df):
    wkd_num_stat = df.groupby(['weekday_num']).agg({'distance_miles': ['mean', 'std'], 'tr_time_nano': ['mean', 'std']})
    wkd_mean_time = wkd_num_stat.apply(lambda x: nano_to_time_str(x.tr_time_nano['mean']), axis=1)
    wkd_std_time = wkd_num_stat.apply(lambda x: nano_to_time_str(x.tr_time_nano['std']), axis=1)
    return_data = {'mean_time' : wkd_mean_time,  'std_time': wkd_std_time, 
                   'mean_dist': wkd_num_stat.distance_miles['mean'], 'std_dist': wkd_num_stat.distance_miles['std']}
    return return_data

def plot_wkd_stat(df, columns):
    for item in columns:
        if len(item) > 1:
            subplot_num = len(item)
            for index, col in enumerate(item):
                xs = ['Mon', 'Tue', 'Wed', 'Th', 'Fri', 'Sat', 'Sun']
                ys = df[col].to_list()
                plt.subplot(1, subplot_num, index+1)
                y_limit = (np.max(df[col]) + (2.5 * np.std(df[col])))
                plt.xlim(-0.2, 7)
                plt.ylim(0, y_limit)
                plt.plot(xs, ys, marker='o', label=col)
                plt.title(col)
        else:
            col = item[0]
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.grid(axis='x')
            xs = ['Mon', 'Tue', 'Wed', 'Th', 'Fri', 'Sat', 'Sun']
            ys = df[col].to_list()
            y_limit = (np.max(df[col]) + (2.5 * np.std(df[col])))
            plt.xlim(-0.2, 7)
            plt.ylim(0, y_limit)
            for index in range(len(xs)):
                ax.text(index + .1, ys[index] + np.std(ys), ys[index], size=12, rotation=0)

            ax.plot(xs, ys, marker='o', label=col)
            plt.title(col)
        plt.show()

def gen_wkd_ridetype_stats(df):
    ridetype_by_wkd = df.groupby(['weekday_num', 'rideable_type']).agg({'rideable_type': 'count'})
    # N = 3 Number of ridetypes classic, docked, and electric
    N = 3
    
    # getting a list of the raw values to be sorted and operated on
    wkd_cnt_list = [item for item in ridetype_by_wkd['rideable_type']]
    
    # group type counts by day (7 groups)
    cntlist = [wkd_cnt_list[n:n+N] for n in range(0, len(wkd_cnt_list), N)]
    
    # calculate the ridetype pcnts by day
    pcnt_list = []
    for item in cntlist:
        for element in item:
            pcnt_list.append(round(element / sum(item) * 100, 2))
            
    ridetype_list_for_cnt = ['classic_bike_cnt', 'docked_bike_cnt', 'electric_bike_cnt']
    ride_cnt_dict = {}
    cnt_list = [[x for x in wkd_cnt_list[::3]],
                [x for x in wkd_cnt_list[1::3]],
                [x for x in wkd_cnt_list[2::3]]]

    for name, data in zip(ridetype_list_for_cnt, cnt_list):
        ride_cnt_dict[name] = data

    ride_cnt_df = pd.DataFrame(ride_cnt_dict)
    ride_cnt_df = ride_cnt_df.T

    ridetype_list_for_pcnt = ['classic_bike_pcnt', 'docked_bike_pcnt', 'electric_bike_pcnt']
    ride_pcnt_dict = {}
    wkd_pcnt_list = [[x for x in pcnt_list[::3]],
                 [x for x in pcnt_list[1::3]],
                 [x for x in pcnt_list[2::3]]]

    for name, data in zip(ridetype_list_for_pcnt, wkd_pcnt_list):
        ride_pcnt_dict[name] = data

    ride_pcnt_df = pd.DataFrame(ride_pcnt_dict)
    ride_pcnt_df = ride_pcnt_df.T
    
    ridetype_stats_df = pd.concat([ride_cnt_df, ride_pcnt_df]).T
    for item in ridetype_list_for_cnt:
        ridetype_stats_df[item] = ridetype_stats_df[item].astype('int')
    return ridetype_stats_df

def generate_wkd_summary(df):
    # counts data
    wkd_counts_dict = generate_wkd_stats(df)
    wkd_counts_df = pd.DataFrame(wkd_counts_dict)
    wkd_counts_df = wkd_counts_df.reindex(columns= [i for i in range(7)])
    wkd_counts_df = wkd_counts_df.T
    wkd_counts_df = wkd_counts_df.rename(columns={0: 'day_str', 1: 'count'})
    
    # agg data
    wkd_stats_df = pd.DataFrame(gen_wkd_stat(df))
    wkd_stats_df[['mean_dist', 'std_dist']] = wkd_stats_df[['mean_dist', 'std_dist']].apply(lambda x: round(x, 3))
    
    # breakdown by ridetype
    wkd_ridetype_df = gen_wkd_ridetype_stats(df)
    
    result_df = pd.concat([wkd_counts_df, wkd_stats_df, wkd_ridetype_df], axis=1)
    
    return result_df

def gen_wkd_ridetype_stats_final(df):
    N = 21
    ridetype_categories = ['classic_bike', 'docked_bike', 'electric_bike']
    df_wkd_ridetype = df.groupby(['weekday_num', 'rideable_type']).agg({'rideable_type': 'count'})
    df_wkd_ridetype_categories = [df_wkd_ridetype.index[i][1] for i in range(len(df_wkd_ridetype))]
    df_wkd_ridetype_list = [item for item in df_wkd_ridetype['rideable_type']]
    
    if len(df_wkd_ridetype_categories) < N:
        df_wkd_ridetype_categories += [''] * (N - len(df_wkd_ridetype_categories))
        df_wkd_ridetype_list = [item for item in df_wkd_ridetype['rideable_type']]
        df_wkd_ridetype_list += [0] * (N - len(df_wkd_ridetype_list))
        df_wkd_ridetype_tuple_list = [(ridetype, cnt) for ridetype, cnt in
                                 zip(df_wkd_ridetype_categories, df_wkd_ridetype_list)]
        
        ideal_ridetype_categories_list = ridetype_categories * 7
        for index in range(N):
            if df_wkd_ridetype_tuple_list[index][0] != ideal_ridetype_categories_list[index]:
                df_wkd_ridetype_tuple_list.insert(index, (ideal_ridetype_categories_list[index], 0))
                df_wkd_ridetype_tuple_list = df_wkd_ridetype_tuple_list[:-1]
                
        df_wkd_ridetype_list = [df_wkd_ridetype_tuple_list[i][1] for i in range(len(df_wkd_ridetype_tuple_list))]
        
    # group type counts by day (7 groups)
    cntlist = [df_wkd_ridetype_list[n:n+3] for n in range(0, len(df_wkd_ridetype_list), 3)]
    
    # calculate the ridetype pcnts by day
    pcnt_list = []
    for item in cntlist:
        for element in item:
            pcnt_list.append(round(element / sum(item) * 100, 2))
            
    ridetype_list_for_cnt = ['classic_bike_cnt', 'docked_bike_cnt', 'electric_bike_cnt']
    ride_cnt_dict = {}
    cnt_list = [[x for x in df_wkd_ridetype_list[::3]],
                [x for x in df_wkd_ridetype_list[1::3]],
                [x for x in df_wkd_ridetype_list[2::3]]]

    for name, data in zip(ridetype_list_for_cnt, cnt_list):
        ride_cnt_dict[name] = data
        
    ride_cnt_df = pd.DataFrame(ride_cnt_dict)
    ride_cnt_df = ride_cnt_df.T
    
    ridetype_list_for_pcnt = ['classic_bike_pcnt', 'docked_bike_pcnt', 'electric_bike_pcnt']
    ride_pcnt_dict = {}
    wkd_pcnt_list = [[x for x in pcnt_list[::3]],
                     [x for x in pcnt_list[1::3]],
                     [x for x in pcnt_list[2::3]]]
    
    for name, data in zip(ridetype_list_for_pcnt, wkd_pcnt_list):
        ride_pcnt_dict[name] = data
        
    ride_pcnt_df = pd.DataFrame(ride_pcnt_dict)
    ride_pcnt_df = ride_pcnt_df.T
    
    ridetype_stats_df = pd.concat([ride_cnt_df, ride_pcnt_df]).T
    for item in ridetype_list_for_cnt:
        ridetype_stats_df[item] = ridetype_stats_df[item].astype('int')
    return ridetype_stats_df

def percentiles_df(df, index_name, subgroup_name):
    df = df[df[index_name].notnull()]
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    pcnts = []
    for percentile in percentiles:
        pcnts.append(round(np.percentile(df[index_name], percentile), 2))
 
    percentiles_df = pd.DataFrame([pcnts], columns = [percentiles], index = [subgroup_name])
    return percentiles_df

def time_stats(df, subgroup_name):
    index_name = subgroup_name + ' time'
    agg_time = df.agg({'tr_time_nano': ['max', 'mean', 'std']})
    agg_time = agg_time.rename(columns={'tr_time_nano': index_name})
#     agg_time['time_str'] = agg_time['tr_time_nano'].apply(lambda x: nano_to_time_str(x))
    agg_time = agg_time.T
    
    percentile_time = percentiles_df(df, 'tr_time_nano', index_name)
    stats_df = pd.concat([percentile_time, agg_time], axis=1)
    
    for index in stats_df.columns[:15].to_list():
        stats_df = stats_df.rename(columns={index: index[0]})
    
    # `.T` tranpose used for easy column creation
    # will be tranposed again turning it into a row
    stats_df = stats_df.T
    stats_df[index_name] = stats_df[index_name].apply(lambda x: int(x))
    index_name2 = index_name + ' str'
    stats_df[index_name2] = stats_df[index_name].apply(lambda x: nano_to_time_str(x))
    stats_df = stats_df.T
    stats_df['measurement'] = 'time'
    stats_df['subgroup'] = [index_name, index_name2]
    
    return stats_df

# Avg Max, Std, Percentiles
def distance_stats(df, subgroup_name):
    index = subgroup_name + ' dist'
    agg_stats = df.agg({'distance_miles': ['max', 'mean', 'std']})
    agg_stats['distance_miles'] = agg_stats['distance_miles'].apply(lambda x: round(x, 2))
    agg_stats = agg_stats.rename(columns={'distance_miles': index})
    agg_stats = agg_stats.T
    percentile_stats = percentiles_df(df, 'distance_miles', index)
    stats_df = pd.concat([percentile_stats, agg_stats], axis=1)
    
    for index in stats_df.columns[:15].to_list():
        stats_df = stats_df.rename(columns={index: index[0]})
    stats_df['measurement'] = 'distance'
    stats_df['subgroup'] = subgroup_name
    return stats_df

def generate_time_dist_percentile_stats(df, subgroup_name):
    distance_stats = distance_stats(df, subgroup_name)
    time_stats = time_stats(df, subgroup_name)
    percentiles_df = pd.concat([distance_stats, time_stats], axis=0)
    return percentiles_df

def generate_wkd_stats(df, plot=False):
    # (Mon, 0) (Tue, 1) (Wed, 2) (Th, 3) (Fri, 4) (Sat, 5) (Sun, 6)
    weekday_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Th': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    weekday_count = Counter(df['weekday_num']).most_common()
    weekday_dict = {}
    for item in weekday_count:
        day_str = get_key(item[0], weekday_map)
        count = item[1]
        month_pcnt = round(count / len(df) * 100, 2)
        weekday_dict[item[0]] = (day_str, count)

    if plot:
        ys = [weekday_dict[i][1] for i in range(0, 7)]
        xs = [weekday_dict[i][0] for i in range(0, 7)]

        plt.figure(figsize=(12, 8))
        plt.plot(xs, ys)
        plt.show()
        
    return weekday_dict

def generate_wkd_summary(df):
    # counts data
    wkd_counts_dict = generate_wkd_stats(df)
    wkd_counts_df = pd.DataFrame(wkd_counts_dict)
    wkd_counts_df = wkd_counts_df.reindex(columns= [i for i in range(7)])
    wkd_counts_df = wkd_counts_df.T
    wkd_counts_df = wkd_counts_df.rename(columns={0: 'day_str', 1: 'count'})
    
    # agg data
    wkd_stats_df = pd.DataFrame(gen_wkd_stat(df))
    wkd_stats_df[['mean_dist', 'std_dist']] = wkd_stats_df[['mean_dist', 'std_dist']].apply(lambda x: round(x, 3))
    
    # breakdown by ridetype
    wkd_ridetype_df = gen_wkd_ridetype_stats_final(df)
    
    result_list = [wkd_counts_df, wkd_stats_df, wkd_ridetype_df]
    
    result_df = pd.concat([wkd_counts_df, wkd_stats_df, wkd_ridetype_df], axis=1)
    
    return result_df

def generate_percentile_stats(df, subgroup_name):
    distance_percentiles = distance_stats(df, subgroup_name)
    time_percentiles = time_stats(df, subgroup_name)
    percentiles_df = pd.concat([distance_percentiles, time_percentiles], axis=0)
    return percentiles_df

def generate_time_dist_percentile_stats(df, subgroup_name):
    distance_stats = distance_stats(df, subgroup_name)
    time_stats = time_stats(df, subgroup_name)
    percentiles_df = pd.concat([distance_stats, time_stats], axis=0)
    return percentiles_df

def percent_of_dict(df, total_df, name):
    count = len(df)
    total_count = len(total_df)
    pcnt = round(count / total_count * 100, 2)
    pcnt_dict = {'name': name,
                 'count': count,
                 'pcnt': pcnt}
    return pcnt_dict

# def plot_percentiles(data_dict):
#     percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
#     fig, ax = plt.subplots(figsize=(12, 8))
#     plt.xticks(percentiles)
#     plt.grid(axis='x')
#     for item in data_dict:
#         label_str = item
#         ys = data_dict[item]
#         xs = percentiles
#         ax.plot(xs, ys, marker='o', label=label_str)
#     plt.legend()
#     plt.show()
    
def plot_dist_percentiles(data_dict):
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xticks(percentiles)
    plt.grid(axis='x')
    for i, item in enumerate(data_dict):
        label_str = item
        color = colors[i]
        ys = data_dict[item].iloc[0, :-5].to_list()
        y_mean = data_dict[item].iloc[0, -4]
        xs = percentiles
        plt.axhline(y=y_mean, color=color, linestyle='--', label=f'{item} mean')
        ax.plot(xs, ys, marker='o', label=label_str)
    plt.legend()
    plt.show()
    
# def plot_time_percentiles(data_dict):
#     percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
#     fig, ax = plt.subplots(figsize=(12, 8))
#     plt.xticks(percentiles)
#     plt.grid(axis='x')
#     for item in data_dict:
#         label_str = item
#         ys = data_dict[item]
#         xs = percentiles
#         ax.plot(xs, ys, marker='o', label=label_str)
#     plt.legend()
#     plt.show()

def plot_time_percentiles(data_dict):
    time_ticks_num = [     3e11,      9e11,    1.8e12,    3.6e12,    5.4e12,    7.2e12,      9e12,   1.08e13]
    time_ticks_str = ['0:05:00', '0:15:00', '0:30:00', '1:00:00', '1:30:00', '2:00:00', '2:30:00', '3:00:00']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xticks(percentiles)
    plt.yticks(time_ticks_num, time_ticks_str)
    plt.grid(axis='x')
    for i, item in enumerate(data_dict):
        label_str = item
        color = colors[i]
        ys = data_dict[item].iloc[1, :-5].to_list()
        y_mean = data_dict[item].iloc[1, -4]
        y_mean_label = data_dict[item].iloc[2, -4]
        xs = percentiles
        plt.axhline(y=y_mean, color=color, linestyle='--', label=f'{item} mean')
        ax.plot(xs, ys, marker='o', color=color, label=label_str)
    plt.legend()
    plt.show()
    
    
def generate_overall_summary_dict(df, subgroup_name):
    # use for months and then aggregated df
    casual = df[df['member_casual'] == 'casual']
    member = df[df['member_casual'] == 'member']
    leisure = leisure_df_filter(df)
    non_leisure = nonleisure_df_filter(df)
    non_events = nonevent_df_filter(df)
    subgroups = {'overall': df, 'casual': casual, 'member': member, 'leisure': leisure, 'non_leisure': non_leisure}
    summary_dict = {}
    subgroup_wkd_dict = {}
    cnt_pcnt_dict = {}
    percentile_dict = {}
    for item in subgroups:
        entry_name = f'{item} {subgroup_name}'
        sub_df = subgroups[item]
        cnt_pcnt_dict[item] = percent_of_dict(sub_df, df, entry_name)
        subgroup_wkd_dict[item] = generate_wkd_summary(sub_df)
        percentile_dict[item] = generate_percentile_stats(sub_df, entry_name)
        
    summary_dict['cnt_pcnt_stats'] = cnt_pcnt_dict
    summary_dict['wkd_dict'] = subgroup_wkd_dict
    summary_dict['percentile_dict'] = percentile_dict
    
    return summary_dict

"""Exclusive use for the report notebook"""
def plot_distance_percentiles(summary_dict):
    data_dict = {'overall': summary_dict['percentile_dict']['overall'], 
                 'member' : summary_dict['percentile_dict']['member'],
                 'casual' : summary_dict['percentile_dict']['casual']}
    
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.xticks(percentiles)
    plt.grid(axis='x')
    for i, item in enumerate(data_dict):
        label_str = item
        color = colors[i]
        ys = data_dict[item].iloc[0, :-5].to_list()
        y_mean = data_dict[item].iloc[0, -4]
        xs = percentiles
        plt.axhline(y=y_mean, color=color, linestyle='--', label=f'{item} mean')
        ax.plot(xs, ys, marker='o', label=label_str)
        
    plt.suptitle("Distance Traveled Percentiles By User Type", fontsize=18)
    plt.title("Percentile distribution of distances between starting and ending coordinates filtered by user type")
    plt.xlabel('Percentiles')
    plt.ylabel('Miles')
    plt.figtext(0.5, 0.01, "Casual Riders travel further than Member Riders over almost the entire distribution", ha="center", fontsize=12)
    plt.legend()
    plt.show()
    
def plot_travel_time_percentiles(summary_dict):
    data_dict = {'overall': summary_dict['percentile_dict']['overall'], 
                 'member' : summary_dict['percentile_dict']['member'],
                 'casual' : summary_dict['percentile_dict']['casual']}
    time_ticks_num = [     3e11,      9e11,    1.8e12,    3.6e12,    5.4e12,    7.2e12,      9e12,   1.08e13]
    time_ticks_str = ['0:05:00', '0:15:00', '0:30:00', '1:00:00', '1:30:00', '2:00:00', '2:30:00', '3:00:00']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.xticks(percentiles)
    plt.yticks(time_ticks_num, time_ticks_str)
    plt.grid(axis='x')
    for i, item in enumerate(data_dict):
        label_str = item
        color = colors[i]
        ys = data_dict[item].iloc[1, :-5].to_list()
        y_mean = data_dict[item].iloc[1, -4]
        y_mean_label = data_dict[item].iloc[2, -4]
        xs = percentiles
        plt.axhline(y=y_mean, color=color, linestyle='--', label=f'{item} mean')
        ax.plot(xs, ys, marker='o', color=color, label=label_str)
        
    plt.suptitle("Travel Time Percentiles By User Type", fontsize=18)
    plt.title("Percentile distribution of time passed from the start to the end of the ride filtered by user type")
    plt.xlabel('Percentiles')
    plt.ylabel('Time')
    plt.figtext(0.5, 0.01, "Casual Riders use Bikeshare longer than Member Riders over the entire distribution", ha="center", fontsize=12)
    plt.legend()
    plt.show()
    
def plot_usertype_count_month(data_dict):
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.grid(axis='x')
    x_labels = ['Jan \'21', 'Feb \'21', 'Mar \'21', 'Apr \'21', 'May \'21', 'Jun \'21', 'Jul \'21', 'Aug \'21', 'Sep \'21', 'Oct \'21', 'Nov \'21', 'Dec \'21', 'Jan \'22']
    for item in data_dict:
        label_str = item
        ys = [data_dict[item][i] for i in data_dict[item]]
        ax.plot(x_labels, ys, marker='o', label=label_str)
    plt.suptitle("User Type Counts By Month", fontsize=18)
    plt.title("Counts of different user types filted by month from January 2021 to January 2022")
    plt.xlabel('Month')
    plt.ylabel('# of Users')
    plt.figtext(0.5, 0.01, "Bikeshare users increase during the warmer seasons", ha="center", fontsize=12)
    plt.legend()
    plt.show()
    
def plot_weekday_count_month(data_dict):
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.grid(axis='x')
    x_labels = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
    for item in data_dict:
        label_str = item
        ys = data_dict[item]
        ax.plot(x_labels, ys, marker='o', label=label_str)
    plt.suptitle("Counts for Day of the Week by Month", fontsize=18)
    plt.title("Counts of users for each day of the week by month from January 2021 to January 2022")
    plt.xlabel('Month')
    plt.ylabel('# of Users')
    plt.figtext(0.5, 0.01, "Bikeshare users ride on the weekends more during the warmer seasons", ha="center", fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.125, 1.01))
    plt.show()