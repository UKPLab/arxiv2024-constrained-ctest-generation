import numpy as np

def split_features(features: np.array, use_bert: bool=False) -> (np.array, np.array):
    # Fetches gap_features and the remaining ones
    del_idx = [56,57,58,59,60]
    if use_bert:
        # Make bert features variables as well
        del_idx += [49,50]
    fixed_features = np.delete(features,del_idx)
    var_features = features[del_idx]
    return fixed_features, var_features
    
def split_gap_features(features: np.array, use_bert: bool=False) -> (np.array, np.array):
    # Fetches gap_features and the remaining ones
    del_idx = [56,57,58,59,60]
    if use_bert:
        # Make bert features variables as well
        del_idx += [49,50]
    fixed_features = np.delete(features,del_idx)
    var_features = features[del_idx]
    return fixed_features, var_features
    
def split_position_features(features: np.array) -> (np.array, np.array):
    # Fetches gap_features and the remaining ones
    del_idx = [51,52,53,54] # removed 55 (gap-token-idx)
    fixed_features = np.delete(features,del_idx)
    var_features = features[del_idx]
    return fixed_features, var_features
    
def split_position_gap_features(features: np.array, use_bert: bool=False) -> (np.array, np.array):
    # Fetches gap_features and the remaining ones
    del_idx_gap = [56,57,58,59,60]
    del_idx_pos = [51,52,53,54]
    if use_bert:
        # Make bert features variables as well
        del_idx_gap += [49,50]
    fixed_features = np.delete(features,del_idx_gap+del_idx_pos)
    var_features_gap = features[del_idx_gap]
    var_features_pos = features[del_idx_pos]
    return fixed_features, var_features_gap, var_features_pos
    
