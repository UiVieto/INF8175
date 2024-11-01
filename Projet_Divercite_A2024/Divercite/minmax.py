from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

transposition_table = {}
def get_other_player(my_player: PlayerDivercite, state: GameState) -> PlayerDivercite:
    for player in state.get_players():
        if player.get_id() != my_player.get_id():
            return player

def evaluate_action(player: PlayerDivercite, action: Action, current_state: GameState) -> float:
    """
    Evaluates the action based on some heuristic relevant to the game, e.g., a score difference heuristic.
    """
    next_state = action.get_next_game_state()
    other_player = get_other_player(player, next_state)
    return next_state.scores[player.get_id()] - next_state.scores[other_player.get_id()]


def minimax_heuristic(player: PlayerDivercite, current_state: GameState, depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf'), remaining_time: int = 1e9):
    """
    Perform a minimax-like recursive heuristic with alpha-beta pruning to find the best action based on future predictions.
    Uses iterative deepening, memoization, and move ordering to improve efficiency.

    Args:
        player (PlayerDivercite): The current player making the move.
        current_state (GameState): The current game state.
        depth (int): The depth of recursion (e.g., 4).
        is_maximizing (bool): Whether to maximize (current player) or minimize (opponent).
        alpha (float): The alpha value for alpha-beta pruning (default: negative infinity).
        beta (float): The beta value for alpha-beta pruning (default: positive infinity).
        remaining_time (int): Remaining time in milliseconds (default: effectively unlimited).

    Returns:
        Tuple[Optional[Action], float]: The best action and its associated score.
    """
    # Check for cached result in transposition table
    state_key = (hash(current_state), depth, is_maximizing)
    if state_key in transposition_table:
        print(f"Using cached result for state {state_key}")
        return transposition_table[state_key]

    if depth == 0 or current_state.is_done():
        other_player = get_other_player(player, current_state)
        score = current_state.scores[player.get_id()] - current_state.scores[other_player.get_id()]
        return None, score

    best_action = None

    # Sort actions for move ordering
    actions = sorted(
        current_state.generate_possible_heavy_actions(),
        key=lambda x: evaluate_action(player, x, current_state),  # Evaluation heuristic for sorting
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

def compute_action(player: PlayerDivercite, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
    """
    Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

    Args:
        current_state (GameState): The current game state.

    Returns:
        Action: The best action as determined by minimax.s
    """
    

    best_action, _ = minimax_heuristic(player, current_state, depth=3, is_maximizing=True, alpha=float('-inf'), beta=float('inf'))
    return best_action
    