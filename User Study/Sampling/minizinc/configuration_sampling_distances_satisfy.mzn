% Description of the ILP problem to sample a set of 4 configurations with maximal distance.
% NOTE: we tackle this as a satisfaction problem; so any feasible solution is fine.

int: n = 144; % total number of possible configurations
int: m = 32; % total number of configurations
int: y = 8; % number of combinations to select
int: z = 4; % number of configurations per combinations (4)
var int: max_distance; % maximum possible distance 
array[1..n, 1..n] of 0..z: all_distances; % array of all distances between two configurations
array[1..n,1..z] of 0..m-1: combinations; % mapping of all combinations to idx
% Mappings
array[1..m] of 1..8: text_diff; % Mapping of idx to {text,diff}
array[1..m] of 1..8: strategy_diff; % Mapping of idx to {strat,diff}
array[1..m] of 1..16: text_strategy; % Mapping of idx to {text,strat}

% Placeholder for values
array[1..y*z] of var int: text_diff_flat; 
array[1..y*z] of var int: strategy_diff_flat; 
array[1..y*z] of var int: text_strategy_flat; 

% Placeholder for counts
array[1..y*z] of var int: text_diff_counts; 
array[1..y*z] of var int: strategy_diff_counts; 
array[1..y*z] of var int: text_strategy_counts; 

% Decision variables
array[1..y] of var 1..n: x;

% Only sample different combinations
include "alldifferent.mzn";
constraint alldifferent(x);
% Sort instances in x
constraint forall(i in 1..y, j in 1..y where i < j)(x[i] < x[j]);

constraint max_distance = sum([ z * i | i in 1..y-1 ]); 

constraint text_diff_flat = [text_diff[ combinations[ x[i],j ]+1 ] | i in 1..y, j in 1..z];
constraint strategy_diff_flat = [strategy_diff[ combinations[ x[i],j ]+1 ] | i in 1..y, j in 1..z];
constraint text_strategy_flat = [text_strategy[ combinations[ x[i],j ]+1 ] | i in 1..y, j in 1..z];

constraint text_diff_counts = [count(text_diff_flat, j) | j in text_diff_flat];
constraint strategy_diff_counts = [count(strategy_diff_flat, j) | j in strategy_diff_flat];
constraint text_strategy_counts = [count(text_strategy_flat, j) | j in text_strategy_flat];

% Ensure that all combinations are balanced
include "all_equal.mzn";
constraint all_equal(text_diff_counts);
constraint all_equal(strategy_diff_counts);
constraint all_equal(text_strategy_counts);

constraint sum([all_distances[x[i],x[j]] | i in 1..y, j in 1..y where i < j]) = max_distance;
% our optimal solution has z*y values times 4 for (X, Diff) and times 2 for (Text,Strategy)

constraint sum(text_diff_counts) = (z*y*4);
constraint sum(strategy_diff_counts) = (z*y*4);
constraint sum(text_strategy_counts) = (z*y*2);
%constraint (z*y*4) + (z*y*4) + (z*y*2)  = sum(text_diff_counts) + sum(strategy_diff_counts) + sum(text_strategy_counts);

solve satisfy;

%output ["Optimal solution: \(x) with distance: \(obj)\n Maximum possible distance is: \(max_distance) \n Text Diff:\n \(text_diff_flat) \n \(td_tmp) \n Strategy Diff:\n \(strategy_diff_flat) \n \(sd_tmp) \n Text Strategy:\n \(text_strategy_flat) \n \(ts_tmp) \n ------------------ \n "];
