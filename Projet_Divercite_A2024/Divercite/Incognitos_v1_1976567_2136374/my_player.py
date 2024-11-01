from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

# Define transposition table for caching
transposition_table = {}

def get_other_player(my_player: PlayerDivercite, state: GameState) -> PlayerDivercite:
    for player in state.get_players():
        if player.get_id() != my_player.get_id():
            return player

def evaluate_action(player: PlayerDivercite, action: Action, current_state: GameState) -> float:
    next_state = action.get_next_game_state()
    other_player = get_other_player(player, next_state)
    return next_state.scores[player.get_id()] - next_state.scores[other_player.get_id()]

def minimax_heuristic(player: PlayerDivercite, current_state: GameState, depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf'), remaining_time: int = 1e9):
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
        key=lambda x: evaluate_action(player, x, current_state),
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

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        best_action, _ = minimax_heuristic(self, current_state, depth=3, is_maximizing=True, alpha=float('-inf'), beta=float('inf'), remaining_time=remaining_time)
        return best_action
