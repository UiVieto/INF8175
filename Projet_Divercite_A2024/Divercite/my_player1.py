import math
import numpy as np
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from enum import Enum
import threading


# Define the JSON cache file for transposition table
transposition_table = {}
class GamePhase(Enum):
    START = "start"
    MID = "mid"
    END = "end"

def get_game_phase(current_state: GameState, start_g = 0.3, end_g = 0.63) -> GamePhase:
    step = current_state.get_step()
    total_steps = current_state.max_step
    if step < total_steps * start_g:
        return GamePhase.START
    elif step < total_steps * end_g:
        return GamePhase.MID
    else:
        return GamePhase.END



def get_other_player(my_player: PlayerDivercite, state: GameState) -> PlayerDivercite:
    for player in state.get_players():
        if player.get_id() != my_player.get_id():
            return player

def evaluate_action(player: PlayerDivercite, current_state: GameState,  game_phase: GamePhase) -> float:
    other_player = get_other_player(player, current_state)
    return current_state.get_scores()[player.get_id()] - current_state.get_scores()[other_player.get_id()]

def critical_score(game_phase: GamePhase, min_score = -2, max_score = 1):
    return min_score if game_phase in {GamePhase.START, GamePhase.MID} else max_score
    
def minimax_heuristic(player: PlayerDivercite, current_state: GameState, depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf'), remaining_time: int = 1e9, critical=False):
    state_key = hash((hash((current_state)), depth, is_maximizing))
    game_phrase = get_game_phase(current_state)
     
    if state_key in transposition_table:
        return transposition_table[state_key]

    if depth == 0 or current_state.is_done():
        score = evaluate_action(player, current_state, game_phrase)
        
        #w_flag = -3 
        #if score < w_flag:
        #    return None, score
        
        #critical_s = critical_score(game_phrase)
        #if not critical and score < critical_s:
        #    _, score = quiescence_search(player, current_state, 1, is_maximizing, alpha, beta, remaining_time, True)
        return None, score

    actions = sorted(current_state.generate_possible_heavy_actions(),
        key=lambda x: evaluate_action(player, x.get_next_game_state(),game_phrase),
        reverse=is_maximizing)
    
    best_action, best_score = None, float('-inf' if is_maximizing else 'inf')
    for action in actions:
     
        next_state = action.get_next_game_state()
        next_depth = depth - 1 if is_maximizing else depth
        _, score = minimax_heuristic(player, next_state, next_depth, not is_maximizing, alpha, beta, remaining_time, critical)
        
        if is_maximizing:
            if score > best_score:
                best_score, best_action = score, action
            alpha = max(alpha, score)
        else:
            if score < best_score:
                best_score, best_action = score, action
            beta = min(beta, score)
        
        if beta <= alpha:
            break

    transposition_table[state_key] = (best_action, best_score)
    return best_action, best_score

def quiescence_search(player: PlayerDivercite, current_state: GameState, depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf'), remaining_time: int = 1e9, critical=True):
    action = None
    best_score = float('-inf')
    best_score_lock = threading.Lock()
    threads = []

    def search_depth(depth_i):
        nonlocal action, best_score
        best_action, score = minimax_heuristic(player, current_state, depth_i, is_maximizing, alpha, beta, remaining_time, critical)
        with best_score_lock:
            if score > best_score:
                print(f"Depth {depth_i}: New best action and score found: {best_action}, {score}")
                best_score, action = score, best_action
        print(f"Depth {depth_i} completed.")

    # Create and start a thread for each depth level
    for depth_i in range(depth, 0, -1):
        thread = threading.Thread(target=search_depth, args=(depth_i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return action, alpha

def dynamic_depth(game_phrase: GamePhase, depth_min = 3, depth_max = 6):
    if game_phrase == GamePhase.START:
        return depth_min
    elif game_phrase == GamePhase.MID:
        return min((depth_max + depth_min) // 2, 5)
    else:
        return 5
    
class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "ShallowMinMax"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state, remaining_time: int = 1e9, **kwargs) -> Action:
        
        game_phase = get_game_phase(current_state)
        depth = dynamic_depth(game_phase)
        print("Depth:", depth)
        best_action, _ = (quiescence_search if game_phase == GamePhase.END else minimax_heuristic)(
            self, 
            current_state, 
            depth, 
            is_maximizing=True, 
            alpha=float('-inf'), 
            beta=float('inf'), 
            remaining_time=remaining_time,
            )
        
        return best_action
