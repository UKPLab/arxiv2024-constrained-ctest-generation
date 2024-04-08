# Merge *_a.txt and *_b.txt files into a single text file
# Creates a file with 40 gaps and features in total.
import os
import sys
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def load_data(infile: str) -> dict:
    data = {}
    with open(infile,'r') as lines:
        for i, line in enumerate(lines):
            hint = ""
            gap = ""
            features = ""
            try:
                word, hint, gap, features = line.strip().split('\t')
            except ValueError:
                word = line.strip()
            data[i] = {"word":word, 
                       "hint":hint, 
                       "gap":gap, 
                       "features":features
                      }
    return data
    
def get_merged_file(a_file: str, b_file: str) -> dict:
    a_data = load_data(a_file)
    b_data = load_data(b_file)
    data = {}
    num_gaps = 0
    try:
        assert(len(a_data)==len(b_data))
    except AssertionError:
        logging.info(f"Error! {a_file} and {b_file} do not match in length!")
        return {}
    for a_idx, b_idx, a_vals, b_vals in zip(a_data.keys(), b_data.keys(), a_data.values(), b_data.values()):
        try:
            assert(a_idx==b_idx)
        except AssertionError:
            logging.info(f"Index mismatch: {b_idx} and {b_idx}!")
            return {}
        try:
            assert(a_vals["word"]==b_vals["word"])
        except AssertionError:
            logging.info(f"Word mismatch: {a_vals['word']} and {b_vals['word']}!")
            return {}
        # No gap at all
        if a_vals["hint"] == "" and b_vals["hint"] == "":
            data[a_idx] = {"word":a_vals['word'],
                           "hint":a_vals['hint'], 
                           "gap":a_vals['gap'], 
                           "features":a_vals['features']
                          }
        # Gap in a
        elif a_vals["hint"] != "" and b_vals["hint"] == "":
            data[a_idx] = {"word":a_vals['word'],
                           "hint":a_vals['hint'], 
                           "gap":a_vals['gap'], 
                           "features":a_vals['features']
                          }
            num_gaps += 1
        # Gap in b
        elif a_vals["hint"] == "" and b_vals["hint"] != "":
            data[a_idx] = {"word":b_vals['word'],
                           "hint":b_vals['hint'], 
                           "gap":b_vals['gap'], 
                           "features":b_vals['features']
                          }
            num_gaps += 1
        # Gap shift in a or b
        else:
            logging.info(f"Possible indexing error in {a_file} and {b_file} at Index {idx}.")
            logging.info(f"Both files have gaps at words {a_data[idx]['word']} and {b_data[idx]['word']}.")
            return {}
    try:
        assert(num_gaps==40)
    except AssertionError:
        logging.info(f"Error! {a_file} and {b_file} only have {num_gaps} gaps!")
        return {}

    return data
    
    

afolder = sys.argv[1] # Input folder
bfolder = sys.argv[2] # Input folder
outfolder = sys.argv[3] # Output folder

infiles = os.listdir(afolder)

logging.info("Merging files.")

tab_char = ord("\t")

for infile in infiles:
    data = get_merged_file(f"{afolder}/{infile}", f"{bfolder}/{infile}")
    if data == {}:
        logging.info(f"Error for the file {infile}!")
        logging.info("Skipping.")
        continue
    with open(f"{outfolder}/{infile}",'w') as outfile:
        for idx, vals in data.items():
            if vals["hint"] == "":
                outfile.write(f"{vals['word']}\n")
            else:
                outfile.write(f"{chr(tab_char).join([vals['word'],vals['hint'],vals['gap'],vals['features']])}{os.linesep}")
    logging.info(f"{infile} successfully merged.")

logging.info("Done.") 









