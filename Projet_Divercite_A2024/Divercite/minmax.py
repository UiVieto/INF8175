from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError


def get_other_player(my_player: PlayerDivercite, state: GameState) -> PlayerDivercite:
    for player in state.get_players():
        if player.get_id() != my_player.get_id():
            return player


def _maxValue(player: PlayerDivercite, current_state: GameState, a: int, b: int, turns_to_search: int = 2,  remaining_time: int = 1e9) -> tuple[int, Action]:
    if turns_to_search == 0 or current_state.is_done():
        other_player = get_other_player(player, current_state)

        return current_state.scores[player.get_id()] - current_state.scores[other_player.get_id()], None
    
    best_score = -999999
    best_action = None

    for action in current_state.generate_possible_heavy_actions():
        state = action.get_next_game_state()
        score, _ = _minValue(player, state, a, b, turns_to_search - 1, remaining_time)

        if score > best_score:
            best_score = score
            best_action = action
            a = max(a, best_score)

        if best_score > b:
            return best_score, best_action

    return best_score, best_action

def _minValue(player: PlayerDivercite, current_state: GameState, a: int, b: int, turns_to_search: int = 2, remaining_time: int = 1e9) -> tuple[int, any]:
    if turns_to_search == 0 or current_state.is_done():
        other_player = get_other_player(player, current_state)

        return current_state.scores[player.get_id()] - current_state.scores[other_player.get_id()], None
    
    worst_score = 999999
    worst_action = None

    for action in current_state.generate_possible_heavy_actions():
        state = action.get_next_game_state()
        score, _ = _maxValue(player, state, a, b, turns_to_search - 1, remaining_time)

        if score < worst_score:
            worst_score = score
            worst_action = action
            b = min(b, worst_score)

        if worst_score <= a:
            return worst_score, worst_action

    return worst_score, worst_action

def compute_action(player: PlayerDivercite, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
    """
    Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

    Args:
        current_state (GameState): The current game state.

    Returns:
        Action: The best action as determined by minimax.s
    """

    _, best_action = _maxValue(player, current_state, -999999, 999999, current_state.get_step() // 10 + 3)

    return best_action
    