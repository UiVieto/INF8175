import copy
import json
import random
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.heavy_action import HeavyAction
from seahorse.game.light_action import LightAction
from seahorse.game.game_layout.board import Piece  # Import Piece class

# Define transposition table for caching
transposition_table = {}

def get_other_player(my_player: PlayerDivercite, state: GameState) -> PlayerDivercite:
    for player in state.get_players():
        if player.get_id() != my_player.get_id():
            return player

def evaluate_positional_strength(game_state, player_id) -> float:
    """Evaluate the positional strength for a specific player based on the current game state."""
    positional_strength = 0
    board_env = game_state.get_rep().get_env()
    dimensions = game_state.get_rep().get_dimensions()

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            if game_state.in_board((i, j)) and (i, j) in board_env:
                piece = board_env[(i, j)]
                if isinstance(piece, Piece) and piece.get_owner_id() == player_id:
                    neighbors = game_state.get_neighbours(i, j)
                    strength = 0
                    for neighbor in neighbors.values():
                        neighbor_piece = neighbor[0]
                        if isinstance(neighbor_piece, Piece):  # Ensure neighbor_piece is a Piece instance
                            if neighbor_piece.get_owner_id() == player_id:
                                strength += 1  # Friendly neighbor
                            elif neighbor_piece.get_type()[0] == piece.get_type()[0]:
                                strength += 0.5  # Matching color/type
                            else:
                                strength -= 0.5  # Opposing player with different type/color
                    positional_strength += strength
    return positional_strength

def determine_game_phase(state: GameState) -> str:
    """Determine game phase based on the number of moves played."""
    if state.get_step() < 10:
        return "start"
    elif 10 <= state.get_step() < 15:
        return "middle"
    else:
        return "end"

def aggressive_heuristic(player: PlayerDivercite, action: Action, next_state: GameState) -> float:
    """Aggressive heuristic: prioritize high immediate gains over the opponent."""
    other_player = get_other_player(player, next_state)
    return next_state.scores[player.get_id()] * 1.2 - next_state.scores[other_player.get_id()] * 0.8

def defensive_heuristic(player: PlayerDivercite, action: Action, next_state: GameState) -> float:
    """Defensive heuristic: focus on minimizing potential loss."""
    other_player = get_other_player(player, next_state)
    return next_state.scores[other_player.get_id()]  * 0.8 - next_state.scores[player.get_id()] * 1.2

def positional_heuristic(player: PlayerDivercite, action: Action, next_state: GameState) -> float:
    """Positional heuristic: assess board control or favorable positions."""
    return evaluate_positional_strength(next_state, player.get_id())

def combined_heuristic(player: PlayerDivercite, action: Action, current_state: GameState, depth: int, game_phase: str) -> float:
    """Combine different heuristics based on game phase."""
    heuristic_weights = {
        "start": {"aggressive": 0.5, "defensive": 0.2, "positional": 0.3},
        "middle": {"aggressive": 0.2, "defensive": 0.5, "positional": 0.4},
        "end": {"aggressive": 0.2, "defensive": 0.7, "positional": 0.1},
    }
    weights = heuristic_weights[game_phase]
    print(game_phase)
    print(weights)
 
    next_state = action.get_next_game_state()
    score = aggressive_heuristic(player, action, next_state)
    
    # Adjust weights dynamically for endgame if the score difference is small
    if game_phase == "end" and abs(score) <= 2:
        weights["defensive"] += 0.1  # Prioritize defense in close games
        weights["positional"] -= 0.1  # Reduce positional influence slightly

    agg_score = aggressive_heuristic(player, action, next_state) * weights["aggressive"]
    def_score = defensive_heuristic(player, action, next_state) * weights["defensive"]
    pos_score = positional_heuristic(player, action, next_state) * weights["positional"]
    return agg_score + def_score + pos_score

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
        key=lambda x: combined_heuristic(player, x, current_state, depth, game_phase),
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
    def __init__(self, piece_type: str, name: str = "OptimizedMinMax"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state, remaining_time: int = 1e9, **kwargs) -> Action:
        depth = min(4, max(1, remaining_time // 10))  # Dynamic depth adjustment
        print(depth)
        best_action, _ = minimax_heuristic(self, current_state, depth=depth, is_maximizing=True, alpha=float('-inf'), beta=float('inf'), remaining_time=remaining_time)
        return best_action
