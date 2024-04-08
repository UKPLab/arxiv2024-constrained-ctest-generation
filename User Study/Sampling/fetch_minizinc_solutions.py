import itertools
import json
import random
from tqdm import tqdm

from minizinc import Instance, Model, Solver

# Checks the constraint for a single configuration.
# For each 4 selections; each text and each strategy need to appear once,
# and each difficulty needs to appear twice. 
def check_constraints(conf):
    diffs = [x[2] for x in conf]
    texts = [x[0] for x in conf]
    strat = [x[1] for x in conf]
    if diffs.count(0.1) != 2:
        return False
    if diffs.count(0.9) != 2:
        return False
    if len(set(texts)) != 4:
        return False
    if len(set(strat)) != 4:
        return False
    return True
    
# Takes a list of combinations (text x strategy x difficulty),
# two indexes, and target cardinality.
# returns true if target_card = comb[idx1,idx2] for all instances. 
def check_cardinality_single(comb, idx1, idx2, target_card):
    comb_flat = []
    for el in comb:
        comb_flat.extend(el)
    targets = [(x[idx1],x[idx2]) for x in comb_flat] # build target list
    for target in targets:
        if targets.count(target) != target_card:
            #print(f"Error for {target}. Should be {target_card} but is {targets.count(target)}")
            return False
    return True

    
# Check if the selected combination of eight configurations is balanced.
# This means that each (text x strategy) should appear twice
# Each (text x difficulty) and (strategy x difficulty) should appear four times.
def check_cardinality(comb):
    if not check_cardinality_single(comb, 0, 2, 4):
        return False
    if not check_cardinality_single(comb, 1, 2, 4):
        return False
    if not check_cardinality_single(comb, 0, 1, 2):
        return False
    return True


# Distance function for two lists 
def get_distance(list1,list2):
    return sum([int(x not in list2) for x in list1])
    
# Check distance function
assert(get_distance([1,2,3,4],[1,2,3,4]) == 0)
assert(get_distance([1,2,3,4],[1,2,3,5]) == 1)
assert(get_distance([1,2,3,4],[1,2,5,6]) == 2)
assert(get_distance([1,2,3,4],[1,5,6,7]) == 3)
assert(get_distance([1,2,3,4],[5,6,7,8]) == 4)

texts = ["T1","T2","T3","T4"]
strats = ["MIP","SEL","SIZE","GPT"]
diffs = [0.1, 0.9]

text_strategy = {   1:  ("T1",  "GPT"),
                    2:  ("T1",  "MIP"),
                    3:  ("T1",  "SEL"),
                    4:  ("T1",  "SIZE"),
                    5:  ("T2",  "GPT"),
                    6:  ("T2",  "MIP"),
                    7:  ("T2",  "SEL"),
                    8:  ("T2",  "SIZE"),
                    9:  ("T3",  "GPT"),
                    10:  ("T3",  "MIP"),
                    11:  ("T3",  "SEL"),
                    12:  ("T3",  "SIZE"),
                    13:  ("T4",  "GPT"),
                    14:  ("T4",  "MIP"),
                    15:  ("T4",  "SEL"),
                    16:  ("T4",  "SIZE"),
                }

text_difficulty = { 1:  ("T1",  "EZ"),
                    2:  ("T1",  "HD"),
                    3:  ("T2",  "EZ"),
                    4:  ("T2",  "HD"),
                    5:  ("T3",  "EZ"),
                    6:  ("T3",  "HD"),
                    7:  ("T4",  "EZ"),
                    8:  ("T4",  "HD"),
                  }

strategy_diff = {   1:  ("GPT", "EZ"),
                    2:  ("GPT", "HD"),
                    3:  ("MIP", "EZ"),
                    4:  ("MIP", "HD"),
                    5:  ("SEL", "EZ"),
                    6:  ("SEL", "HD"),
                    7:  ("SIZE", "EZ"),
                    8:  ("SIZE", "HD"),
                }
                
diff_mapping = {0.1:"EZ", 0.9:"HD"}

text_strategy_rev = {v:k for k,v in text_strategy.items()}
text_difficulty_rev = {v:k for k,v in text_difficulty.items()}
strategy_diff_rev = {v:k for k,v in strategy_diff.items()}

all_lists = [texts, strats, diffs] # diffs

configurations = list(itertools.product(*all_lists))
permutations = set(itertools.combinations(configurations, 4))
# Final filtered list of instances 
perms_filtered = [x for x in permutations.copy() if check_constraints(x)]
final_permutations = {i+1:x for i,x in enumerate(sorted(list(perms_filtered))) }


print("Num combinations: ", len(configurations)) # 32
print("Num permutations: ", len(permutations)) # 35960

comb_dict = {i:comb for i,comb in enumerate(sorted(configurations))}
comb_rev = {comb:i for i,comb in comb_dict.items()}
    
print("Num permutations filtered: ", len(final_permutations)) # 144

# Generate permutations X permutations array of pairwise distances (144x144) 
distances = []
combinations_mzn = []
for _, list1 in sorted(final_permutations.items()):
    sub_distances = []
    combinations_mzn.append([comb_rev[x] for x in list1])
    for _, list2 in sorted(final_permutations.items()):
        sub_distances.append(get_distance(list1,list2))
    distances.append(sub_distances[:])

# Generate lists of text x strategy, text x difficulties, and strategy x difficulties
text_strategy_mzn = []
text_difficulty_mzn = []
strategy_diff_mzn = []
for idx, combs in sorted(comb_dict.items()):
    t,s,d = combs
    text_strategy_mzn.append(text_strategy_rev[(t,s)])
    text_difficulty_mzn.append(text_difficulty_rev[(t,diff_mapping[d])])
    strategy_diff_mzn.append(strategy_diff_rev[(s,diff_mapping[d])])
    

# Write all data in minizinc format
with open("outputs/all_distances.dzn",'w') as f:
    # Write mappings
    f.write(f"text_strategy = [{','.join([str(x) for x in text_strategy_mzn])}];\n\n")
    f.write(f"text_diff = [{','.join([str(x) for x in text_difficulty_mzn])}];\n\n")
    f.write(f"strategy_diff = [{','.join([str(x) for x in strategy_diff_mzn])}];\n\n")
    # Write permuations:
    f.write("combinations = [")
    for list1 in combinations_mzn:
        f.write(f"| {','.join([str(x) for x in list1])} \n")
    f.write("|];\n\n")
    # Write distances
    f.write("all_distances = [")
    for distance in distances:
        f.write(f"| {','.join([str(x) for x in distance])} \n")
    f.write("|];\n\n")

# Load model from file
model = Model("minizinc/configuration_sampling_distances_satisfy.mzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Gecode Instance of the model 
instance = Instance(gecode, model)
# Assign 4 to n
instance["combinations"] = combinations_mzn
instance["text_strategy"] = text_strategy_mzn
instance["text_diff"] = text_difficulty_mzn
instance["strategy_diff"] = strategy_diff_mzn
instance["all_distances"] = distances
result = instance.solve(all_solutions=True) # fetch all solutions

print(f"Number of possible solutions: {len(result)}")


allowed_combs = []
# Sanity check
for i in tqdm(range(len(result))):
    try:
        if check_cardinality([final_permutations[x] for x in result[i,'x']]):
            allowed_combs.append(result[i])
    except IndexError:
        print(result[i,'x'])

print(len(allowed_combs))

random.seed(42)

final_comb = random.randint(0, len(result))
print(f"Combination to use: {result[final_comb,'x']} with distance {result[final_comb,'max_distance']}")
print(f"text_diff_counts: {result[final_comb,'text_diff_counts']}")
print(f"strategy_diff_counts: {result[final_comb,'strategy_diff_counts']}")
print(f"text_strategy_counts: {result[final_comb,'text_strategy_counts']}")

final_results = {}

for comb in result[final_comb,'x']:
    combination = final_permutations[comb]
    print(combination)
    final_results[comb] = combination

with open("outputs/final_combinations.json",'w') as f:
    json.dump(final_results,f)
    
# example optimial sol:
opt = [1, 23, 49, 71, 77, 122, 94, 144]
print(opt)
for o in opt: 
    combination = final_permutations[o]
    print(combination)





