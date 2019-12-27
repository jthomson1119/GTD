import pandas as pd # data manipulation/storing
import datetime # date handling
import matplotlib.pyplot as plt # plotting
# %matplotlib inline 

import numpy as np # data manipulation, calculations
import pickle # saving/loading
# from pandas_ods_reader import read_ods

import operator # for sorting dicts
from collections import defaultdict # dict with default val, used for top counts below

import time # timing operations

import os
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

# preprocessing
def date_obj_var(row):
    """returning datetime obj"""
    return datetime.datetime(row['iyear'],row['imonth'],row['iday'])


def ret_weekday(row):
    """returning weekday"""
    return row['date_obj'].weekday()


def gtd_add_date_obj(df):
    """replacing bad month dates, adding datetime obj + weekday"""
    # replacing zero mons
    print 'df preprocess shape: {}'.format(df.shape)
    df['imonth'] = df['imonth'].replace(0,1)
#     df['imonth'].value_counts()
    df['iday'] = df['iday'].replace(0,1)
    df['date_obj'] = df.apply(date_obj_var,axis=1)
    df['weekday'] = df.apply(ret_weekday,axis=1)
    print 'df postprocess shape: {}'.format(df.shape)
#     return df
    
def load_plus_dates(path):
    df = pd.read_csv(path)
    gtd_add_date_obj(df)
    print 'unique event ids == to row count? {}'.format(df['eventid'].nunique() == df.shape[0])
    return(df)

def get_vars(filepath):
    var_list = pd.read_csv(filepath)
    return var_list


def top_value_count(df,feat):
    top_val = df[feat].value_counts().index[0]
    top_count = df[feat].value_counts()[0]
    return top_val, top_count

def check_feats(df,feat):
    print df[feat].isna().sum()
    print 'number -9 or -99: {}'.format()
    
    
def feat_desc(df,feat,quant=0):
    print 'feat : {}'.format(feat)
    na_sum = df[feat].isna().sum()
    na_pct = float(na_sum) / df.shape[0] * 100.
    neg9 = df[df[feat] == -9].shape[0]
    neg99 = df[df[feat] == -99].shape[0]
    unknown_sum = (neg9 + neg99)
    unknown_pct = float(unknown_sum) / df.shape[0] * 100.
    
    print 'na sum: {} and pct: {}'.format(na_sum, na_pct)
    print 'unknown sum: {} and pct: {}'.format(unknown_sum,unknown_pct)
    if quant == 0:
        nunique = df[feat].nunique()
        print 'number unique: {}'.format(nunique)
        
        
        
# eda / describing
def top_vals(df,cat,n_return=0,n_print=0):
    '''return, print (or both or none) of categorical value counts on dataframe.
    n_return = -1 for all, if > len(val_counts) will return all.
    n_print for number of vals to print.'''
    val_counts = df[cat].value_counts()
    index = val_counts.index
    if n_print > 0:
        for i in range(n_print):
            try:
                val = index[i]
                count = val_counts[val]
                per = float(count) / df.shape[0] * 100.
                print 'value: {} has {:,} count ({:.2f} %)'.format(val, count, per)
            except:
                pass
    if n_return < 0:
        return val_counts
    elif n_return > 0:
        if n_return > len(val_counts):
            n_return == len(val_counts)
        return val_counts[:n_return]
    
def casualties_by_x(df, by_column, head_number=-1, ret=0):
    print '{} has {:,} casualties: {:,} deaths ({:.2f}%) and {:,} wounds ({:.2f}%)'.format(
        'Total', 
        df['nkill'].sum() + df['nwound'].sum(),
#         int(row['nkill'] + row['nwound']), 
        df['nkill'].sum(),
        float(df['nkill'].sum()) / float(df['nkill'].sum() + df['nwound'].sum()) * 100.,
        df['nwound'].sum(),
        float(df['nwound'].sum()) / float(df['nkill'].sum() + df['nwound'].sum()) * 100.)
    casualties_by = df.groupby([by_column])['nkill','nwound'].sum()
    casualties_by['casualties'] = casualties_by['nkill'] + casualties_by['nwound']
    print '-'*30
    for item, row in casualties_by.sort_values(by=['casualties'], ascending=False).head(head_number).iterrows():
        print '{} has caused {:,} casualties: {:,} deaths ({:.2f}%) and {:,} wounds ({:.2f}%)'.format(
        row.name, 
        int(row['casualties']),
#         int(row['nkill'] + row['nwound']), 
        int(row['nkill']),
        float(row['nkill']) / row['casualties'] * 100.,
        int(row['nwound']),
        float(row['nwound']) / row['casualties'] * 100.)
        print '-'*30
    if ret == 1:
        return casualties_by.sort_values(by=['casualties'],ascending=False).head(head_number)

def islam_holiday_total(holiday_dict,df):
    '''returning basic stats for major islamic holidays.'''
    holiday_totals = {}
    for k_year, v_dict in holiday_dict.items():
        if k_year >= 1970:
            yearly_holidays = {}
#             print k_year
            ashura = v_dict["Fasting 'Ashura"]
#             print 'ashura: {}'.format(ashura)
            new_year = v_dict['Islamic New Year']
#             print 'new year: {}'.format(new_year)
            hajj_start = v_dict["Wuquf in 'Arafa (Hajj)"]
            hajj_end = v_dict['Eid ul-Adha']
            
            ramadan_start = v_dict['Start of Fasting Ramadan']
            ramadan_end = v_dict['Eid ul-Fitr']
            ashura_data = holiday_stats(df,[ashura])
            new_year_data = holiday_stats(df,[new_year])
            hajj_data = holiday_stats(df,[hajj_start,hajj_end])
            ramadan_data = holiday_stats(df,[ramadan_start,ramadan_end])
            yearly_holidays['ramadan'] = ramadan_data
            yearly_holidays['ashura'] = ashura_data
            yearly_holidays['new_year'] = new_year_data
            yearly_holidays['hajj'] = hajj_data
            holiday_totals[k_year] = yearly_holidays
#         print '\n'
#         print '-'*30
    return holiday_totals

def holiday_stats(df,holiday_dates,extended=0):
    '''expecting holiday_dates to be list (even if individual)...
    extended = n_days to extend ELSE zero'''
    holiday_data = {}
    # changing dates based on extended
    if extended > 0:
        if len(holiday_dates) > 1:
            holiday_dates[0] -= datetime.timedelta(days=extended)
            holiday_dates[1] += datetime.timedelta(days=extended)
        else:
            holiday_dates[0] -= datetime.timedelta(days=extended)
            holiday_dates.append(holiday_dates[0] + datetime.timedelta(days=extended*2))
    # creating MASK, used to get indices
    if len(holiday_dates) > 1:
        mask = (df['date_obj'] >= holiday_dates[0]) & (df['date_obj'] <= holiday_dates[1])
    else:
        mask = df['date_obj'] == holiday_dates[0]
    indices = df.loc[mask].index
    n_attacks = len(indices)
    n_kills = df.loc[indices]['nkill'].sum()
    n_wounds = df.loc[indices]['nwound'].sum()
    n_suicides = df.loc[indices]['suicide'].sum()
    
    holiday_data['indices'] = indices
    holiday_data['n_attacks'] = n_attacks
    holiday_data['n_kills'] = n_kills
    holiday_data['n_wounds'] = n_wounds
    holiday_data['n_suicides'] = n_suicides
    return holiday_data

def bloodiest_streak_PROD(df):
    ''' pass in FULL df, returning dict containing...............................'''
    df.sort_values('date_obj',inplace=True)
    streak_max = 0
    streaks = {}
#     check = 0
    streak_included = []
    
    for i, (idx,row) in enumerate(df.iterrows()):
#         print 'contents of prev streak: \n'
#         print streak_included
#         print i, idx, row['eventid'], row['date_obj']
        # if index in prev streak, pass.....
#         print 'IS IDX IN PREV STREAK????? ---- ',idx in streak_included
        if idx not in streak_included:
#             pass
        
            streak_data = {}
            index_list = []
            streak_n = 0
            while streak_n < df.shape[0] - i - 1:

                beg = list(df['date_obj'])[i+streak_n]
                end = list(df['date_obj'])[i+streak_n+1]
                next_idx = list(df.index.values)[i+streak_n+1]
                days_btw = (end-beg).days
#                 print '{} ... to ... {} ...... {} .............{}'.format(beg,end,days_btw,next_idx)
                if days_btw > 1:
                    break
                streak_n += 1
                index_list.append(next_idx)
    #         print idx, row['date_obj']
            # if streak occured:
            if streak_n > 1:
    #             streak_included.append(index_list)
                streak_included += index_list
                streak_included.append(idx)
#                 print 'len streak: {}'.format(streak_n)
#                 check += 1
                index_list.insert(0,idx)
#                 print 'len index_list: {}'.format(len(index_list))
                streak_data['count'] = streak_n
                streak_data['indices'] = index_list
                nkills = df.loc[index_list]['nkill'].sum()
                nwounds = df.loc[index_list]['nwound'].sum()
                streak_data['n_kill'] = nkills
                streak_data['n_wound'] = nwounds
                streaks[idx] = streak_data
    #         if i >= 30:
    #             break
#         print '-'*100
    
#     df.sort_index(inplace=True)
    return streaks


def group_stat_collect_PROD(df,feats,cats_using,quants_using,dums_using,streak_finder=1):
    # loading in holiday_dict
    holiday_dict = pickle.load(open('data/holidays','rb'))
    
    org_stats = {}
    cats = {}
    quants = {}
    dums = {}
#     holidays = {}
    streaks = {}
    dates = {}
        
    n_obs = df.shape[0]
    date_min = df['date_obj'].min()
    date_max = df['date_obj'].max()
    days_operating = (date_max - date_min).days
    days_per_event = days_operating / float(n_obs)
    events_per_day = float(n_obs) / days_operating
    is_active = 1 if df['iyear'].max() >= 2015 else 0
    
    dates['is_active'] = is_active
    dates['earliest_date'] = date_min
    dates['latest_date'] = date_max
    dates['days_operating'] = days_operating
    dates['total_events'] = n_obs
    dates['events_per_day'] = events_per_day
    
    print 'GROUP ACTIVE ??: {}'.format(bool(is_active))
#     print is_active
    print 'nobs: {} .. days: {} .... event per day: {}'.format(n_obs, days_operating, events_per_day) # , days_per_event
    print 'events per day: {} ........ days per event: {}'.format(events_per_day, days_per_event)
    print 'dates: min {} and max {}'.format(date_min, date_max)
    for feat in feats:
        if feat in cats_using:
#             print 'feat: {}'.format(feat)
            cats[feat] = top_vals(df,feat,-1,0)
        elif feat in quants_using:
#             print 'feat: {}'.format(feat)
            quants[feat] = df[feat].describe()
#             org_stats[feat] = df[feat].describe()
        elif feat in dums_using:
#             print 'feat: {} mean: {}'.format(feat,df[feat].mean())
            dums[feat] = df[feat].mean()
#             org_stats[feat] = df[feat].mean()
#             print df[feat].mean()
    holidays = islam_holiday_total(holiday_dict,df)
#     print 'holiday type (in groupstatsdev): {}'.format(type(holidays))
    if streak_finder:
        streaks = bloodiest_streak_PROD(df)
    else:
        streaks = {}
    org_stats['dates'] = dates
    org_stats['cats'] = cats
    org_stats['quants'] = quants
    org_stats['dums'] = dums
    org_stats['holidays'] = holidays
    org_stats['streaks'] = streaks
    return org_stats


# def holiday_meta_stats(holiday_dict):
#     islam_holiday_casualties = 0
#     islam_holiday_attacks = 0
#     for year, holiday_data in holiday_dict.items():
#         for holiday, data in holiday_data.items():
#             islam_holiday_casualties += data['n_kills'] + data['n_wounds']
#             islam_holiday_attacks += data['n_attacks']
#     return islam_holiday_casualties, islam_holiday_attacks

# def streak_meta_data(streak_dict):
#     days_in_streaks = 0
#     streak_casualties = 0
#     streak_num = len(streak_dict)
#     longest_streak = len(max(streak_dict.iteritems(), key=operator.itemgetter(1))[1]['indices'])
    
#     for idx, data in streak_dict.items():
#         days_in_streaks += len(data['indices'])
#         streak_casualties += data['n_kill'] + data['n_wound']
#     return days_in_streaks, streak_casualties, streak_num, longest_streak

def streak_meta_data_PROD(streak_dict):
#     print len(streak_dict)
#     print type(streak_dict)
    streak_meta_dict = {}
    if len(streak_dict) < 1:
        streak_num = np.nan
        days_in_streaks = np.nan
        longest_streak = np.nan
        streak_casualties = np.nan
    else:
        days_in_streaks = 0
        streak_casualties = 0
        streak_num = len(streak_dict)
        longest_streak = len(max(streak_dict.iteritems(), key=operator.itemgetter(1))[1]['indices'])

        for idx, data in streak_dict.items():
            days_in_streaks += len(data['indices'])
            streak_casualties += data['n_kill'] + data['n_wound']
    streak_meta_dict['number_of_streaks'] = streak_num
    streak_meta_dict['days_in_streaks'] = days_in_streaks
    streak_meta_dict['longest_streak'] = longest_streak
    streak_meta_dict['streak_casualties'] = streak_casualties
    return streak_meta_dict
#     return days_in_streaks, streak_casualties, streak_num, longest_streak

def holiday_meta_stats_PROD(holiday_dict):
    holiday_casualties = 0
    holiday_attacks = 0
    ramadan_attacks = 0
    ramadan_casualties = 0
    for year, holiday_data in holiday_dict.items():
        for holiday, data in holiday_data.items():
            holiday_casualties += data['n_kills'] + data['n_wounds']
            holiday_attacks += data['n_attacks']
            if holiday == 'ramadan':
                ramadan_attacks += data['n_attacks']
                ramadan_casualties += data['n_kills'] + data['n_wounds']
    holiday_meta_dict = {}
    holiday_meta_dict['holiday_casualties'] = holiday_casualties
    holiday_meta_dict['holiday_attacks'] = holiday_attacks
    holiday_meta_dict['ramadan_attacks'] = ramadan_attacks
    holiday_meta_dict['ramadan_casualties'] = ramadan_casualties
    return holiday_meta_dict
#     return islam_holiday_casualties, islam_holiday_attacks, ramadan_attacks, ramadan_casualties

def cat_meta_stats(cat_dict):
    '''passing in cat_dict --> key == cat feat, val == val counts (as series)'''
    cat_meta_dict = {}
#     print len(cat_dict)
#     print type(cat_dict)
#     print 'keys: {}'.format(cat_dict.keys())
#     print type(cat_dict['city'])
    for k_feat, v_values in cat_dict.items():
#         print '{} '.format(k_feat)
#         print v_values[0], v_values.index[0], sum(v_values)
#         print '{} accounts for {} %'.format(v_values.index[0], (float(v_values[0]) / sum(v_values) * 100.))
        cat_meta_dict['{}_most_used'.format(k_feat)] = v_values.index[0]
        cat_meta_dict['{}_pct'.format(k_feat)] = float(v_values[0]) / sum(v_values) * 100.
    return cat_meta_dict
#         print

def quant_meta_stats(quant_dict):
    '''passing in cat_dict --> key == cat feat, val == val counts (as series)'''
    quant_meta_dict = {}
#     print len(quant_dict)
#     print type(quant_dict)
#     print quant_dict.keys()
#     print type(quant_dict['nwound'])
    for k_feat, v_values in quant_dict.items():
#         print k_feat
#         print 'max: {}'.format(v_values['max'])
#         print 'mean: {}'.format(v_values['mean'])
#         print 'sum: {}'.format(v_values['mean'] * v_values['count'])
        quant_meta_dict['{}_avg'.format(k_feat)] = round(v_values['mean'],2)
        quant_meta_dict['{}_max'.format(k_feat)] = v_values['max']
        quant_meta_dict['{}_sum'.format(k_feat)] = v_values['mean'] * v_values['count']
    return quant_meta_dict


# modeling
def values_list_generator(df,col,num_deep=0):
    col_idx = df[col].value_counts().index
#     le = preprocessing.LabelEncoder()
    if num_deep == 0:
        val_list = [val if val in col_idx else 'other' for val in df[col]]
    else:
        val_list = [val if val in col_idx[:num_deep] else 'other' for val in df[col]]
#     df[str(col) + '_raw'] = val_list
#     df[str(col) + '_encoded'] = le.fit_transform(val_list)
#     return df
    return val_list

def check_create_model_dir(dir_path='models'):
    '''check if model dir in path, else creates'''
    if dir_path in os.listdir(os.curdir):
        print 'directory available'
    else:
        os.makedirs(dir_path)
        print 'new dir created: {}'.format(dir_path in os.listdir(os.curdir))

def gbc_model(X_train, y_train, param_grid, n_estimators=100, cv_num=5):
    '''gradient boost model creation and train'''
    gbc = GradientBoostingClassifier(n_estimators=n_estimators)
    gbc_grid = GridSearchCV(estimator=gbc,param_grid=param_grid,cv=cv_num)
    gbc_grid.fit(X_train,y_train)
    print 'best parameters: {} \ntraining score: {}'.format(gbc_grid.best_params_, gbc_grid.best_score_)
    return gbc_grid

def separate_preds_from_total(df_og,gname_encoding='',pred_ratio=.6):
    '''separate "unknown" gnames by pred_ratio, 
    returning original df with (1 - pred_ratio) of "unknowns" and
    preds df with pred_ratio of "unknowns"'''
    unknowns = df_og[df_og['gname_encoded'] == 'Unknown']
#     print 'og shape: {}..... and unkn shape: {}'.format(df_og.shape, unknowns.shape) 
    preds = unknowns.sample(n = int(np.round(unknowns.shape[0] * pred_ratio)))
#     print 'preds shape: {}'.format(preds.shape)
    return df_og.drop(index=preds.index), preds

def yearly_model(yearly_df, param_grid, model_path, n_deep = 4, test_size = .4):
    '''
    1) create gname_encoded_list / add to df
    2) preprocess df - drop "gname" (since target = gname_encoded) and "iyear" (since modeling by year)
    3) split original df into training and preds
    4) dropping eventid- no info contained
    5) train model'''
    # making copy of df
    df = yearly_df.copy()
    gname_encoded_list = values_list_generator(yearly_df,'gname',n_deep)
#     print 'len encoded gnames: {}'.format(len(gname_encoded_list)) # debug
    df['gname_encoded'] = gname_encoded_list
    # deleting gname
    df.drop('gname',axis=1,inplace=True)
    # dropping 'iyear'
    df.drop('iyear',axis=1,inplace=True)
    # splitting up modeling vs preds
    df_model, df_preds = separate_preds_from_total(df)
    # discarding eventid from model
    df_model.drop('eventid',axis=1,inplace=True)
    # pred eventids to ser: --- will use later
    pred_eventids = df_preds['eventid']
    # dropping pred eventids
    df_preds.drop(['eventid','gname_encoded'],axis=1,inplace=True)
#     df_preds.drop('')
#     print 'model shape: {} and pred shape: {}'.format(df_model.shape, df_preds.shape)
    df_y = df_model['gname_encoded']
    df_model.drop('gname_encoded',axis=1,inplace=True)
    
    # train test splits:
    X_train, X_test, y_train, y_test = train_test_split(df_model,
                                                   df_y,
                                                   test_size = test_size,
                                                   stratify=df_y)
#     print 'x train shape: {} and y train shape: {}\n\
#     x test shape: {} y test shape: {}'.format(X_train.shape, y_train.shape, 
#                                               X_test.shape, y_test.shape)
    # time to model
    gbc_grid = gbc_model(X_train,y_train, param_grid)
    test_accuracy = accuracy_score(y_test,gbc_grid.predict(X_test))
    # printing accuracy
    print 'test accurcay is: %f' % test_accuracy
    
    # saving model
    pickle.dump(gbc_grid, open(model_path,'wb'))
    
    
#     return df_preds
    # predictions on 'left out' unknowns...
    new_labels = gbc_grid.predict(df_preds)
    # return new_labels
    df_preds['eventid'] = pred_eventids
    df_preds['new_label'] = new_labels
#     df_preds['gname_old'] = 
    return df_preds, gbc_grid.best_score_, test_accuracy
