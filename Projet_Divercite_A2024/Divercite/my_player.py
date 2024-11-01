import copy
import json
import random
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.heavy_action import HeavyAction
from seahorse.game.light_action import LightAction

# Define transposition table for caching
transposition_table = {}

def get_other_player(my_player: PlayerDivercite, state: GameState) -> PlayerDivercite:
    for player in state.get_players():
        if player.get_id() != my_player.get_id():
            return player

def determine_game_phase(state: GameState) -> str:
    # Determine game phase based on the number of moves played
    if state.get_step() < 20:
        return "start"
    elif 20 <= state.get_step() < 50:
        return "middle"
    else:
        return "end"

def depth_weight(depth: int, game_phase: str) -> float:
    # Assign weights based on depth and game phase
    if game_phase == "start":
        weights = {1: 0.2, 2: 0.3, 3: 0.5}  # Emphasize shallow depths for exploration
    elif game_phase == "middle":
        weights = {1: 0.3, 2: 0.3, 3: 0.4}  # Balanced weights for mid-game
    else:  # end
        weights = {1: 0.2, 2: 0.4, 3: 0.4}  # Prioritize deeper levels for accurate endgame moves
    
    return weights.get(depth, 1.0)  # Default to 1.0 if depth is beyond predefined weights

def evaluate_action(player: PlayerDivercite, action: Action, current_state: GameState, depth: int, game_phase: str) -> float:
    next_state = action.get_next_game_state()
    other_player = get_other_player(player, next_state)
    base_score = next_state.scores[player.get_id()] - next_state.scores[other_player.get_id()]
    weight = depth_weight(depth, game_phase)
    return base_score * weight

def minimax_heuristic(player: PlayerDivercite, current_state: GameState, depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf'), remaining_time: int = 1e9):
    game_phase = determine_game_phase(current_state)
    state_key = (hash(current_state), depth, is_maximizing)
    if state_key in transposition_table:
        return transposition_table[state_key]

    if depth == 0 or current_state.is_done():
        other_player = get_other_player(player, current_state)
        score = current_state.scores[player.get_id()] - current_state.scores[other_player.get_id()]
        return None, score

    best_action = None
    actions = sorted(
        current_state.generate_possible_heavy_actions(),
        key=lambda x: evaluate_action(player, x, current_state, depth, game_phase),
        reverse=is_maximizing
    )

    if is_maximizing:
        max_score = float('-inf')
        for action in actions:
            state = action.get_next_game_state()
            _, score = minimax_heuristic(player, state, depth - 1, False, alpha, beta, remaining_time)
            if score > max_score:
                max_score = score
                best_action = action
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        transposition_table[state_key] = (best_action, max_score)
        return best_action, max_score
    else:
        min_score = float('inf')
        for action in actions:
            state = action.get_next_game_state()
            _, score = minimax_heuristic(player, state, depth - 1, True, alpha, beta, remaining_time)
            if score < min_score:
                min_score = score
                best_action = action
            beta = min(beta, score)
            if beta <= alpha:
                break
        transposition_table[state_key] = (best_action, min_score)
        return best_action, min_score

class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "ShallowMinMax"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state, remaining_time: int = 1e9, **kwargs) -> Action:
        # Determine the optimal depth based on current game state and remaining time
        best_action, _ = minimax_heuristic(self, current_state, depth=4, is_maximizing=True, alpha=float('-inf'), beta=float('inf'), remaining_time=remaining_time)
        return best_action
