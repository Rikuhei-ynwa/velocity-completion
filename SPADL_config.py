# -*- coding: utf-8 -*-
"""Configuration of the SPADL language.

Attributes
----------
FIELD_LENGTH : float
    The length of a pitch (in meters).
FIELD_WIDTH : float
    The width of a pitch (in meters).

"""
from typing import List

import numpy as np
import pandas as pd  # type: ignore
import re

FOOTBALL_PLAYER_NUM = 11
SUBSTITUTION_NUM = 3

FIELD_LENGTH: float = 105.0  # unit: meters
FIELD_WIDTH: float = 68.0  # unit: meters
FIELD_LENGTH_ORIGINAL: float = 120.0  # unit: meters
FIELD_WIDTH_ORIGINAL: float = 80.0  # unit: meters

DEFENSE_TYPE = [
    "defensive_ball_recovery",
    "defensive_block",
    "defensive_foul",
    "clearance",
    'tackle',
    "interception",
    "own_goal",
    "shield",
]

KEEPER_SPECIFIC_TYPE = ["keeper_save","keeper_claim","keeper_punch","keeper_collect",]

# Define functions to create Metrica data formats, but follow the StatsBomb definitions
def create_Type(row) -> str:
    # Ball Receipt
    # if row['type_name'] == "Ball Receipt":
    #     return "ball_receipt"
    # Ball Recovery
    if row['type_name'] == "Ball Recovery":
        if row["possession_team_id"] == row["team_id"]:
            return "offensive_ball_recovery"
        else:
            return "defensive_ball_recovery"
    # Block
    elif row['type_name'] == "Block":
        if row["possession_team_id"] == row["team_id"]:
            return "offensive_block"
        else:
            return "defensive_block"
    # Carry
    elif row['type_name'] == "Carry":
        return "carry"
    # Clearance
    elif row['type_name'] == "Clearance":
        return "clearance"
    # Dispossessed
    elif row['type_name'] == "Dispossessed":
        return "dispossessed"
    # Dribble
    elif row['type_name'] == "Dribble":
        return "dribble"
    # Duel
    elif row['type_name'] == "Duel":
        if row['sub_type_name'] == "Tackle":
            return "tackle"
        else:
            return np.nan
    # Foul Commited
    elif row['type_name'] == "Foul Committed":
        if row["possession_team_id"] == row["team_id"]:
            return "offensive_foul"
        else:
            return "defensive_foul"
    # Goal Keeper
    elif row['type_name'] == "Goal Keeper":
        if "Save" in row['sub_type_name']:
            return "keeper_save"
        elif row['outcome_name'] == 'Claim':
            return "keeper_claim"
        elif row['sub_type_name'] == 'Punch':
            return "keeper_punch"
        elif row['sub_type_name'] == 'Collected':
            return "keeper_collect"
        elif row['sub_type_name'] == 'Smother':
            return "tackle"
        else:
            return np.nan
    # Interception
    elif row['type_name'] == "Interception":
        return "interception"
    # Miscontrol
    elif row['type_name'] == "Miscontrol":
        return "miscontrol"
    # Pass (to calculate OBSO, we want to know the height of a pass)
    elif row['type_name'] == "Pass":
        if row["pass_height_name"] == "High Pass":
            return "high_pass"
        elif row["pass_height_name"] == "Low Pass":
            return "low_pass"
        elif row["pass_height_name"] == "Ground Pass":
            return "ground_pass"
    # Own Goal Against
    elif row['type_name'] == 'Own Goal Against':
        return "own_goal"
    # Shield
    elif row['type_name'] == 'Shield':
        return "shield"
    # Shot
    elif row['type_name'] == "Shot":
        if row['sub_type_name'] == 'Penalty':
            return "penalty_shot"
        elif row['sub_type_name'] == 'Penalty':
            return "free_kick_shot"
        elif row['sub_type_name'] == 'Corner':
            return "corner_shot"
        else:
            return "shot"
    else:
        return np.nan
    

def create_Subtype(row) -> str:
    # Ball Receipt
    # if row['type_name'] == "Ball Receipt":
    #     if row['outcome_name'] == 'Incomplete':
    #         return "fail"
    #     else:
    #         return "success"
    # Ball Recovery
    if row['type_name'] == "Ball Recovery":
        if 'ball_recovery_recovery_failure' in row and row['ball_recovery_recovery_failure']:
            return "fail"
        else:
            return "success"
    # Block
    elif row['type_name'] == "Block":
        if 'block_offensive' in row and row['block_offensive']:
            return "fail"
        else:
            return "success"
    # Carry
    elif row['type_name'] == "Carry":
        return "success"
    # Clearance
    elif row['type_name'] == "Clearance":
        return "success"
    # Dispossessed
    elif row['type_name'] == "Dispossessed":
        return "fail"
    # Dribble
    elif row['type_name'] == "Dribble":
        if row['outcome_name'] == 'Complete':
            return "success"
        elif row['outcome_name'] == 'Incomplete':
            return "fail"
        else:
            return np.nan
    # Duel
    elif row['type_name'] == "Duel":
        if row['outcome_name'] in ['Won','Success','Success In Play','Success Out']:
            return "success"
        elif row['outcome_name'] in ['Lost','Lost Out','Lost In Play']:
            return "fail"
        else:
            return np.nan
    # Foul Commited
    elif row['type_name'] == "Foul Committed":
        if 'foul_committed_advantage' in row and row['foul_committed_advantage']:
            return np.nan
        else:
            return "fail"
    # Goal Keeper
    elif row['type_name'] == "Goal Keeper":
        if row['outcome_name'] in [
            'Claim','Clear','Collected Twice','In Play Safe','Saved Twice','Success','Touched Out','Won','Success In Play','Success Out','Punched Out'
            ]:
            return "success"
        elif row['outcome_name'] in ['Fail','In Play Danger','No Touch','Touched In','Lost In Play','Lost Out']:
            return "fail"
        else:
            return np.nan
    # Interception
    elif row['type_name'] == "Interception":
        if row['outcome_name'] in ['Success','Success In Play','Success Out']:
            return "success"
        elif row['outcome_name'] in ['Lost','Lost In Play','Lost Out']:
            return "fail"
        else:
            return np.nan
    # Miss Control
    elif row['type_name'] == "Miscontrol":
        return "fail"
    # Own Goal Against
    elif row['type_name'] == 'Own Goal Against':
        return "fail"
    # Pass
    elif row['type_name'] == "Pass":
        if row['outcome_name'] in ['Incomplete','Out']:
            return "fail"
        elif row['outcome_name'] in ['Injury Clearance','Unknown']:
            return np.nan
        elif row['outcome_name'] == 'Pass Offside':
            return "offside"
        else:
            return "success"
    # Shield
    elif row['type_name'] == "Shield":
        return "success"
    # Shot
    elif row['type_name'] == "Shot":
        if row['outcome_name'] == 'Goal':
            return "success"
        else:
            return "fail"
    else:
        return np.nan
    

def create_Time(row,half_end_second) -> float:
    if row['period'] == 1:
        return row['minute'] * 60 + row['second']
    elif row['period'] == 2:
        return half_end_second + (row['minute'] - 45) * 60 + row['second']
    else:
        return np.nan


def create_Coordinates(coordinate, side) -> float:
    if side == "x":
        return (coordinate / FIELD_LENGTH_ORIGINAL) * FIELD_LENGTH - (FIELD_LENGTH / 2)
    elif side == "y":
        return (FIELD_WIDTH - (coordinate / FIELD_WIDTH_ORIGINAL) * FIELD_WIDTH) - (FIELD_WIDTH / 2)


