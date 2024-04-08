import os
import csv
import numpy as np
import string
import random
##################
# Table functions
##################

def read_ctest(infile):
    data = {}
    with open(infile,'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            data[int(row[0])] = {"ctest":row[1], 
                                 "gaps":row[2],
                                 "plaintext":row[3],
                                 "tokens":row[4],
                                 "type":row[5]
                                 }
                            
    return data

def read_ctest_user_mapping(infile):
    data = {}
    with open(infile,'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            data[int(row[0])] = {"ctest_id":int(row[1]), 
                                 "user_id":int(row[2]),
                                 "responses":row[3],
                                 "errors":[int(x) for x in row[4].split()],
                                 "score":int(row[5].split('/')[0]),
                                 "time":row[6]
                                 }
                                
    return data

def read_questionnaire(infile):
    data = {}
    with open(infile,'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            data[int(row[0])] = {"user_id":int(row[1]), 
                                 "cefr":row[2],
                                 "years":int(row[3]),
                                 "frequency":row[4],
                                 "native":row[5],
                                 "other_lang":row[6],
                                 "other_lang_list":row[7]
                                 }
                            
    return data

def get_random_keys(n):
    keys = set()
    while len(keys) < n:
        # get random string of length 6 without repeating letters
        result_str = ''.join(random.sample(string.ascii_lowercase, 8))
        keys.add(result_str)
    return list(keys)

def read_user(infile):
    keys = get_random_keys(40)
    data = {}
    with open(infile,'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for k, row in zip(keys, reader):
            data[int(row[0])] = {"strategy":int(row[5]),
                                 "key":k}
                                
    return data
    
##################
# Helper functions
##################


def get_time_in_seconds(timestring):
    # Gets timestring (time_taken) from the study and returns the total time
    # in seconds. time_taken is formatted as : hours:minutes:seconds.miliseconds
    hours, mins, secs = timestring.split(':')
    total_time = float(hours)*3600 + float(mins)*60 + float(secs)
    return total_time

def get_time_in_seconds_max(timestring, cap):
    # Gets timestring (time_taken) from the study and returns the total time
    # in seconds. time_taken is formatted as : hours:minutes:seconds.miliseconds
    hours, mins, secs = timestring.split(':')
    total_time = float(hours)*3600 + float(mins)*60 + float(secs)
    if total_time > cap:
        return cap
    return total_time

def get_text_difficulty(dictkey):
    # fetches the ctest-type and returns the model, text, and tau
    model, text_diff = dictkey.split()
    text = text_diff.replace('.txt','')[:-4]
    diff = float(text_diff.replace('.txt','')[-3:])
    return model.replace('baseline_','').replace('_ctest_xgb',''), text, diff
    
def get_tau(score):
    return 1.0 - (float(score)/20.0)

##################
#    Analysis
##################
datapath = "study_data_raw"

# This is mean(time) + 5 * std(time)
max_time = 1369.0073668397467

ctest_data = read_ctest(os.path.join(datapath, "ctest.csv"))
ctest_user_data = read_ctest_user_mapping(os.path.join(datapath, "ctest_user_mapping.csv"))
questionnaire_data = read_questionnaire(os.path.join(datapath, "questionnaire.csv"))
user_data = read_user(os.path.join(datapath, "user.csv"))

ctest_types = {}

for idx, vals in ctest_data.items():
    ctest_types[vals['type']] = idx

ctest_types_idx = {v:k for k,v in ctest_types.items()}

with open('r_data/data.csv','w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f,delimiter=',')
    tableheader = ["index","difficulty","time","user_key","user_idx","model","tau","text","cefr","years","frequency"]
    writer.writerow(tableheader)
    idx = 0
    for user_idx, user_vals in user_data.items():
        user_questionnaire = [qvals for _, qvals in questionnaire_data.items() if qvals["user_id"] == user_idx][0]
        for ctest_idx, ctest_vals in sorted(ctest_user_data.items()):
            if user_idx != ctest_vals['user_id']:
                continue
            model, text, diff = get_text_difficulty(ctest_data[ctest_vals['ctest_id']]['type'])
            writer.writerow([idx, get_tau(ctest_vals['score']), get_time_in_seconds_max(ctest_vals['time'], max_time), user_vals['key'], user_idx, model, diff, text, user_questionnaire['cefr'], user_questionnaire['years'], user_questionnaire['frequency']])
            idx += 1

