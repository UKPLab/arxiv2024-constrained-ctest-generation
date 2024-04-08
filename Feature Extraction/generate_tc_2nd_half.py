import os
import sys

def read_ctest_tc(infile):
    data = {"ctest":"",
             "solution":"",
             "tokens":""
             }
    ctest = []
    solution = []
    tokens = []
    with open(infile) as lines:
        for line in lines:
            try:
                tken, idx, hint, err = line.strip().split('\t')
                ctest.append(f"{hint}#GAP#")
                tokens.append(tken)                
            except ValueError:
                ctest.append(line.strip())
                tokens.append(line.strip())
    data["ctest"] = " ".join(ctest)
    data["solution"] = " ".join(solution)
    data["tokens"] = " ".join(tokens)
    return data

def write_tc_file(out_1st, out_2nd, ctest):
    first_sent = True
    even = True
    test = ctest['ctest'].strip().split()
    gaps = ctest['solution'].strip().split()
    tokens = ctest['tokens'].strip().split()
    gap_idx = 0
    with open(f"{out_1st}",'w') as out1, open( f"{out_2nd}",'w') as out2:
        for tes, tok in zip(test,tokens):
            if gap_idx > 19:
                first_sent = True
            if '-' in tok:
                out1.write(f"{tok}\n")
                out2.write(f"{tok}\n")
                continue
            if tok in ["``","''"]:
                out1.write(f"{tok}\n")
                out2.write(f"{tok}\n")
                continue
            if tok in ['.','?','!']:
                out1.write(f"{tok}\n")
                out2.write(f"{tok}\n")
                first_sent = False
                continue
            if first_sent:
                out1.write(f"{tok}\n")
                out2.write(f"{tok}\n")
                continue
            if len(tok) == 1:
                out1.write(f"{tok}\n")
                out2.write(f"{tok}\n")
                continue
            gap_len = int(len(tok)/2)

            if even:
                out1.write(f"{tok}\n")
                out2.write(f"{tok}\t{gap_idx+1}\t{tok[:gap_len]}\t0.5\n")
            else:
                try:
                    assert("#GAP#" in tes)
                    out2.write(f"{tok}\n")
                    out1.write(f"{tok}\t{gap_idx+1}\t{tok[:gap_len]}\t0.5\n")
                    gap_idx += 1 
                except AssertionError:
                    out1.write(f"{tok}\n")
                    out2.write(f"{tok}\n")
                    continue

            even = not even

infolder = sys.argv[1]
first = sys.argv[2]
second = sys.argv[3]

for infile in os.listdir(infolder):
    dat = read_ctest_tc(os.path.join(infolder, infile))
    write_tc_file(os.path.join(first, infile), os.path.join(second, infile), dat)
    
    
