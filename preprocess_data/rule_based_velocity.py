"""Rule-based player velocities of the previous event-based OBSO study (the rule-based baseline of the paper).

The rules come from the earlier study on StatsBomb 360 freeze frames, which contain no player velocities
(https://github.com/Rikuhei-ynwa/DRSO, pitch control with assigned velocities), applied to the event types of
event_config.EVENT_TYPE. In the data sets they are stored in rows 47..68 of "cies" (docs/DATA_SCHEMA.md). For every
player with a valid position (checked in the order given):

  actor                         the ball velocity of the event if the event carries the ball (dribble, carry),
                                otherwise zero
  opponent nearest to the actor 5 m/s towards the end position of the ball
  defending goalkeeper          for shots: 2 m/s along the goal line towards the end position of the ball
                                (sign of ball_end_y - gk_y), otherwise zero. The defending goalkeeper is the
                                goalkeeper of the actor's team for defensive actions (foul, clearance, tackle,
                                interception, block, keeper_save), otherwise the goalkeeper of the other team
  every other player            5 m/s towards the centre of the goal attacked by the actor's team
Players without a valid position or velocity get zero.
"""

import numpy as np

import event_config as cfg


def _toward(target, pos, speed):
    d = np.asarray(target, float) - np.asarray(pos, float)
    return d / np.linalg.norm(d) * speed


def rule_based_velocities(event_type, pos, valid, actor, gk_home, gk_away, ball_end, ball_vel, attack_dir_home):
    """Rule-based velocities of the 22 players of one event.

    event_type      str, one of event_config.EVENT_TYPE
    pos             (22, 2) positions [m]; players 0..10 = home, 11..21 = away
    valid           (22,) bool, players with a valid position and velocity
    actor           index (0..21) of the player on the ball
    gk_home/gk_away indices of the goalkeepers
    ball_end        (2,) end position of the ball [m]
    ball_vel        (2,) ball velocity [m/s] (NaN if unknown)
    attack_dir_home +1 if the home team attacks towards +x, -1 otherwise
    """
    home = np.arange(11)
    actor_home = actor in home
    opponents = np.arange(11, 22) if actor_home else home
    dist = np.linalg.norm(pos[opponents] - pos[actor], axis=1)
    opponent_near_actor = int(opponents[np.argmin(dist)])
    defending = event_type in cfg.DEFENSE_TYPE or event_type in cfg.KEEPER_SPECIFIC_TYPE
    defending_gk = (gk_home if actor_home else gk_away) if defending else (gk_away if actor_home else gk_home)
    attack_dir = attack_dir_home if actor_home else -attack_dir_home
    goal = np.array([attack_dir * cfg.FIELD_LENGTH / 2, 0.0])

    v = np.zeros((22, 2))
    for j in range(22):
        if not valid[j]:
            continue
        if j == actor:
            if event_type in cfg.CARRYING_TYPE and np.all(np.isfinite(ball_vel)):
                v[j] = ball_vel
        elif j == opponent_near_actor:
            v[j] = _toward(ball_end, pos[j], cfg.ABS_FAKE_VELOCITY)
        elif j == defending_gk:
            if "shot" in event_type:
                v[j] = [0.0, 2.0] if ball_end[1] >= pos[j, 1] else [0.0, -2.0]
        else:
            v[j] = _toward(goal, pos[j], cfg.ABS_FAKE_VELOCITY)
    return v
