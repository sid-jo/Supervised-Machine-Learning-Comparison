# Importing data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

fruits = pd.read_csv('data/fruits/citrus.csv')

chess = pd.read_csv('data/chess/chess_games.csv')

music = pd.read_csv('data/music/pop_classical.csv')

lep_columns = ['poisonous', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 
           'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 
           'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
           'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
           'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
lepiota = pd.read_csv('data/mushroom/agaricus-lepiota.data', names = lep_columns)

################################ FRUITS ##########################################

# Converting targets to boolean
# 1 - orange, -1 - grapefruit
def fruit_bool(val):
    if val == 'orange':
        return 1
    else:
        return -1

fruits = fruits.assign(name = fruits['name'].apply(fruit_bool))

def load_fruits():
    # Dropping weight column
    fruits.drop(['weight'], axis = 1, inplace = True)
    
    # Convert data into feature (X) and label (Y) arrays
    fruit_X = fruits.drop(['name'], axis = 1).to_numpy()
    fruit_Y = fruits['name'].to_numpy()

    print('Fruit feature space:', fruit_X.shape) # 10000 x 4
    print('Fruit label space', fruit_Y.shape) # 10000 x 1
    
    return fruit_X, fruit_Y

################################ CHESS ##########################################

# Converting values to booleans
# rated: 1 - yes, 0 - no
# winner: 1 - white, -1 - black
def chess_bool(val):
    if not val:
        return 0
    elif val == 'Black':
        return -1
    else:
        return 1

def load_chess():
    
    # Removing unnecessary variables
    chess_cleaned = chess.drop(['game_id', 'time_increment', 'white_id', 'black_id', 'moves', 'opening_moves', 'opening_fullname',
            'opening_shortname', 'opening_response', 'opening_variation'], axis = 1)
    
    chess_cleaned = chess_cleaned.assign(rated = chess_cleaned['rated'].apply(chess_bool))
    chess_cleaned = chess_cleaned.assign(winner = chess_cleaned['winner'].apply(chess_bool))

    # One-hot encoding all categorical features
    ohe = OneHotEncoder(sparse = False)
    chess_X = ohe.fit_transform(chess_cleaned.get(['victory_status', 'opening_code']))

    # Convert data into feature (X) and label (Y) arrays
    num_features = chess_cleaned.get(['rated', 'turns', 'white_rating', 'black_rating']).to_numpy()

    # Combining numerical features array and one-hot encoded features
    chess_X = np.hstack((num_features, chess_X))
    chess_Y = chess_cleaned['winner'].to_numpy()

    print('Chess feature space:', chess_X.shape) # 20058 x 373
    print('Chess label space', chess_Y.shape) # 20058 x 1
    
    return chess_X, chess_Y
######################################### MUSIC ##########################################

# Setting labels to booleans
# Classical - 1, Pop - 0
def label_bool(val):
    if val == 2:
        return 1
    else:
        return -1

def load_music():
    # Dropping filename column
    music_cleaned = music.drop(['filename'] , axis = 1)
    music_cleaned = music_cleaned.assign(label = music_cleaned['label'].apply(label_bool))

    # Since there are no categorical variables, no one-hot encoding is required
    # Convert data into feature (X) and label (Y) arrays

    music_X = music_cleaned.drop(['label'], axis = 1).to_numpy()
    music_Y = music_cleaned['label'].to_numpy()

    print('Music feature space:', music_X.shape) # 200 x 28
    print('Music label space', music_Y.shape) # 200 x 1
    
    return music_X, music_Y

################################# LEPIOTA ##########################################

# Defining transformation functions
def ring_num(val):
    if val == 'n':
        return 0
    elif val == 'o':
        return 1
    else:
        return 2

def to_bool(val):
    if val == 't' or val == 'p':
        return 1
    elif val == 'f':
        return 0
    else:
        return -1

def pop_trans(val):
    if val in['a', 'n', 'v']:
        return 'm' 
    else:
        return val

def habitat_trans(val):
    if val == 'g':
        return 'm'
    elif val == 'w':
        return 'u'
    elif val in ['l', 'p']:
        return 'd'
    else:
        return val

def load_lepiota():
    
    # Dropping certain features to simplify data set and remove any missing values
    lep_cleaned = lepiota.drop(['gill_attachment', 'gill_spacing', 'gill_color', 'stalk_root', 
              'stalk_surface_above_ring', 'stalk_surface_below_ring', 
              'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_color',
              'ring_type', 'spore_print_color'], axis = 1)
    
    # Applying transformations
    lep_cleaned = lep_cleaned.assign(ring_number = lep_cleaned.get('ring_number').apply(ring_num))
    lep_cleaned = lep_cleaned.assign(bruises = lep_cleaned.get('bruises').apply(to_bool))
    lep_cleaned = lep_cleaned.assign(poisonous = lep_cleaned.get('poisonous').apply(to_bool))
    lep_cleaned = lep_cleaned.assign(population = lep_cleaned.get('population').apply(pop_trans))
    lep_cleaned = lep_cleaned.assign(habitat = lep_cleaned.get('habitat').apply(habitat_trans))

    # Convert data into feature (X) and label (Y) arrays
    lep_X = lep_cleaned.drop(['poisonous'], axis = 1) # Features
    lep_Y = lep_cleaned.get('poisonous').to_numpy() # Labels

    # One-hot encoding selected feature values
    ohe = OneHotEncoder(sparse = False)
    lep_X = ohe.fit_transform(lep_X) 

    print('Lepiota feature space:', lep_X.shape) # 8124 x 46
    print('Lepiota label space:', lep_Y.shape) # 8124 x 1

    return lep_X, lep_Y