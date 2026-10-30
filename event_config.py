# -*- coding: utf-8 -*-
"""Constants of the vendor-neutral intermediate format (docs/DATA_SCHEMA.md).

Only the constants that the models, the preprocessing and the evaluation use are kept here. The
provider-specific settings of the original research code (column names and coordinate systems of the
delivered data) are not part of the release.
"""

FOOTBALL_PLAYER_NUM = 11
FIELD_PLAYERS_NUM = 22

FIELD_LENGTH: float = 105.0  # unit: meters
FIELD_WIDTH: float = 68.0  # unit: meters
ATTCKING_THIRD = FIELD_LENGTH / 6  # target events: ball start and end beyond this distance from the centre line

FPS = 25  # frame rate of the frame numbers stored with every event

PLAYERS_SPEED_LIMIT = 10.5  # m/s; 10.5 * 3.6 = 37.8 km/h

# rule-based velocities (preprocess_data/rule_based_velocity.py)
ABS_FAKE_VELOCITY = 5.0  # m/s
CARRYING_TYPE = ["dribble", "carry"]
DEFENSE_TYPE = ["foul", "clearance", "tackle", "interception", "block"]
KEEPER_SPECIFIC_TYPE = ["keeper_save"]

# padding of positions outside the pitch (missing players, events before the start of a half)
MISSING_CIE_X = (-1) * FIELD_LENGTH * 2
MISSING_CIE_Y = (-1) * FIELD_WIDTH * 2

# the 18 event-type categories of the one-hot event-type feature (order matters)
EVENT_TYPE = [
    'pass',
    'cross',
    'throw_in',
    'freekick_crossed',
    'freekick_short',
    'corner_crossed',
    'corner_short',
    'foul',
    'tackle',
    'interception',
    'shot',
    'shot_penalty',
    'shot_freekick',
    'keeper_save',
    'clearance',
    'non_action',  # padding
    'dribble',
    'goalkick',
]

# rows of the "cies" array: 24 locations, 23 velocities, 22 rule-based velocities
LOC_NODES = [
    'ball_start', 'ball_end',
    'Home_1', 'Home_2', 'Home_3', 'Home_4', 'Home_5', 'Home_6', 'Home_7', 'Home_8', 'Home_9', 'Home_10', 'Home_11',
    'Away_1', 'Away_2', 'Away_3', 'Away_4', 'Away_5', 'Away_6', 'Away_7', 'Away_8', 'Away_9', 'Away_10', 'Away_11'
]
VEL_NODES = [
    'ball',
    'Home_1', 'Home_2', 'Home_3', 'Home_4', 'Home_5', 'Home_6', 'Home_7', 'Home_8', 'Home_9', 'Home_10', 'Home_11',
    'Away_1', 'Away_2', 'Away_3', 'Away_4', 'Away_5', 'Away_6', 'Away_7', 'Away_8', 'Away_9', 'Away_10', 'Away_11'
]
VEL_FAKE_NODES = [
    'Home_1', 'Home_2', 'Home_3', 'Home_4', 'Home_5', 'Home_6', 'Home_7', 'Home_8', 'Home_9', 'Home_10', 'Home_11',
    'Away_1', 'Away_2', 'Away_3', 'Away_4', 'Away_5', 'Away_6', 'Away_7', 'Away_8', 'Away_9', 'Away_10', 'Away_11'
]

NUM_NODE_FEATURES = 5
NUM_EDGE_FEATURES = 2

# match-level split (number of matches)
NUM_TRAINING_SET = 40
NUM_VALIDATION_SET = 8
NUM_TEST_SET = 7
