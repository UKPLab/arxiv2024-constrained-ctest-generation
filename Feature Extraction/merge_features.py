import sys
import os

def read_dkpro_file(infile):
    data = {}
    with open(infile,'r') as lines:
        start_data = False
        name = ""
        feature_names = []
        fname_index = 5 # Index of the file and gapname
        bad_files = set()
        for line in lines:
            if line.strip() == "":
                continue
                
            if not start_data and line.strip() != "@data":
                fname = line.strip().split()[1]

                if fname == "outcome":
                    continue
                    
                if fname == "DKProTCInstanceID":
                    continue
                    
                feature_names.append(fname)
                
            if start_data:
                splits = line.strip().split(',')
                tmp_name = splits[fname_index]
                name = tmp_name.split('.txt')[0] + '.txt'
                if name in bad_files:
                    continue
                tmp_idx = tmp_name.split('.txt_')[-1]
                tmps = tmp_idx.split('_')
                gap_idx = tmps[0]
                gap = '_'.join(tmps[1:])
                #gap_idx, gap = tmp_idx.split('_')
                del splits[-1] # remove score tag
                del splits[fname_index] # remove filename
                try:
                    assert(len(splits) == 59) # We expect 59 features
                except AssertionError:
                    print("Error in file {}".format(tmp_name))
                    print("Expected 59 features but got {}".format(len(splits)))
                    bad_files.add(name)
                    continue
                try:
                    data[name][int(gap_idx)] = {'gap':gap,
                                                'features':splits[:]}
                except KeyError:
                    data[name] = {int(gap_idx):{'gap':gap,
                                                'features':splits[:]}}
                                                
            if "@data" in line:
                start_data = True
            
        # Data cleanup
        delete_counter = 0
        for badname in list(bad_files):
            try:
                del data[badname[1:]]
                delete_counter += 1
            except KeyError:
                continue
        if delete_counter > 0:
            print("Removed {} faulty files.".format(delete_counter))
                
    return data, feature_names
    
def add_bert_features(bertfile, data):
    with open(bertfile,'r') as lines:
        filename = bertfile.strip().split('/')[-1]
        for i, line in enumerate(lines):
            gap, pred_prob, entropy = line.strip().split('\t')
            assert(data[filename][i]['gap'] == gap)
            data[filename][i]['features'].append(pred_prob)
            data[filename][i]['features'].append(entropy)
    return data


       
def merge_tc_data(tc_file, outpath, data):
    with open(tc_file,'r') as lines, open(outpath,'w') as outlog:
        filename = tc_file.strip().split('/')[-1]
        try:
            for line in lines:
                if len(line.strip().split('\t')) > 1:
                    token, idx, hint, score = line.split('\t')
                    gap_idx = int(idx) - 1 # tc files start couting at 1 not 0.
                    assert(data[filename][gap_idx]['gap'] == token)
                    outlog.write("{}\t{}\t{}\t{}\t{}\n".format(token, gap_idx, hint, token.replace(hint,""), ','.join(data[filename][gap_idx]['features'])))
                else:
                    outlog.write(line.strip() + '\n')
        except AssertionError:
            print("Bad tc_file for file {}".format(tc_file))
            return False
    return True
    

feature_file = sys.argv[1]
bert_folder = sys.argv[2]
tc_folder = sys.argv[3]
outfolder = sys.argv[4]

data, feature_names = read_dkpro_file(feature_file)

succ_file = []
tc_fail_file = []
bert_fail_file = []

for tc_file in data.keys():
    try:
        data = add_bert_features(os.path.join(bert_folder, tc_file), data)
    except (FileNotFoundError, AssertionError, KeyError, ValueError) as e:
        bert_fail_file.append(tc_file)
        continue
    if merge_tc_data(os.path.join(tc_folder, tc_file), os.path.join(outfolder,tc_file), data):
        succ_file.append(tc_file)
    else:
        tc_fail_file.append(tc_file)
        
print("Successfully generated {} files".format(len(succ_file)))
print("Features:", feature_names)
print("Issues with the following tc files:")
print(tc_fail_file)
print("Issues with the following BERT files:")
print(bert_fail_file)
    








