from enum import Enum
from itertools import product

from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.game_layout.board import Piece

from player_divercite import PlayerDivercite


ONE_MINUTE = 60
class GamePhase(Enum):
    START = "start"
    MID = "mid"
    END = "end"

Evaluation = tuple[int, float]  # Score difference, Positional strength


class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "HeuristicMinMaxer") -> None:
        """
        Initialize the PlayerDivercite instance.
        
        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "ShallowMinMax")
        """
        super().__init__(piece_type, name)
        self.transposition_table = {}

    def get_game_phase(self, current_state: GameState, start_g = 0.3, end_g = 0.63) -> GamePhase:
        """
        Determine the current game phase based on the current state.
        The game phase is divided into three parts: start, mid, and end.
        
        Args:
            current_state (GameState): Current game state
            start_g (float, optional): Start game phase threshold (default is 0.3)
            end_g (float, optional): End game phase threshold (default is 0.63)
        
        Returns:
            GamePhase: Current game phase
        """
        step = current_state.get_step()
        total_steps = current_state.max_step
        if step < total_steps * start_g:
            return GamePhase.START
        elif step < total_steps * end_g:
            return GamePhase.MID
        else:
            return GamePhase.END
        
    def dynamic_depth(self, game_phrase: GamePhase, remaining_time: int) -> int:
        """ 
        Determines the depth of the search tree based on the game phase.
        
        Args:
            game_phrase (GamePhase): Current game phase
            remaining_time (int): Remaining time for the agent
        
        Returns:
            int: Determined depth of the search tree
        """
        if ONE_MINUTE > remaining_time or GamePhase.START == game_phrase:
            return 3
        elif GamePhase.MID == game_phrase or ONE_MINUTE * 4 > remaining_time:
            return 4
        else:
            return 5
    
    def get_opponent(self, state: GameState) -> PlayerDivercite:
        """
        Get the other player in the game.
        
        Args:
            state (GameState): Current game state
            
        Returns:
            PlayerDivercite: The other player
        """
        players = state.get_players()
        return players[1] if players[0] == self else players[0]
    
    def get_player_score(self, state: GameState, scores: dict[int, float]) -> int:
        """
        Compute the difference in scores between the player and the opponent.
        
        Args:
            state (GameState): Current game state
            scores (dict[int, float]): Scores of the players
            
        Returns:
            int: Score difference between the player and the opponent
        """
        opponent = self.get_opponent(state)
        
        previous_player_score = scores[self.get_id()]
        previous_opponent_score = scores[opponent.get_id()]
        
        player_score = state.scores[self.get_id()]
        opponent_score = state.scores[opponent.get_id()]
        score = player_score - opponent_score
        
        if previous_player_score > player_score:
            score -= 1
        if previous_opponent_score > opponent_score:
            score += 2
        
        return score
    
    def evaluate_positional_strength(self, game_state, player_id) -> float:
        """Evaluate the positional strength for a specific player based on the pieces in the current game state."""
        positional_strength = 0
        board_env = game_state.get_rep().get_env()
        dimensions = game_state.get_rep().get_dimensions()

        for i, j in product(range(dimensions[0]), range(dimensions[1])):
            if game_state.in_board((i, j)) and (i, j) in board_env:
                piece = board_env[(i, j)]
                if not isinstance(piece, Piece) or piece.get_owner_id() != player_id:
                    continue  # Pass if it is not a piece or if it is the piece of the opponent

                neighbors = game_state.get_neighbours(i, j)
                strength = 0.0

                if (i, j) in [(3, 4), (4, 3), (5, 4), (4, 5)]:
                    strength += 0.2  # Bonus for pieces in the center

                for neighbor in neighbors.values():
                    neighbor_piece = neighbor[0]
                    if isinstance(neighbor_piece, Piece):  # Ensure neighbor_piece is a Piece instance
                        if neighbor_piece.get_owner_id() == player_id:
                            if neighbor_piece.get_type()[0] == piece.get_type()[0]:
                                strength += 1.0  # Friendly matching color/type
                            else:
                                strength -= 0.5  # Competition between friendly pieces
                        else:
                            if neighbor_piece.get_type()[0] == piece.get_type()[0]:
                                strength += 1.0  # Possible gain of ressource from opposing player 
                            strength -= 1.0  # Opposing player with different type/color
                positional_strength += strength
        return positional_strength
    
    def evaluate_action(self, current_state: GameState, score: dict[int, float]) -> Evaluation:
        """
        Evaluate the action to be taken.
        
        Args:
            current_state (GameState): Current game state
            game_phase (GamePhase): Current game phase
            
        Returns:
            float: The score of the action
        """
        score_difference = self.get_player_score(current_state, score)
        positional_strength = self.evaluate_positional_strength(current_state, self.get_id())

        return score_difference, positional_strength

    def minmax_heuristic(self, current_state: GameState, remaining_time: int, depth: int, game_ph: GamePhase) -> Action:
        scores_c = current_state.scores

        def max_h(state: GameState, alpha: Evaluation, beta: Evaluation, depth: int) -> tuple[Action, float]:
            """
            Maximizing player logic using Minimax or PVS during the endgame.
            
            Args:
                state (GameState): Current game state
                alpha (int): Alpha value
                beta (int): Beta value
                depth (int): Depth of the search tree
            
            Returns:
                tuple[Action, float]: The best action and its associated score
            """
            state_key = hash((hash(state), depth, True))
            if state_key in self.transposition_table:
                return self.transposition_table[state_key]

            if depth == 0 or state.is_done():
                return None, self.evaluate_action(state, scores_c)

            best_action, max_score = None, (float('-inf'), float('-inf'))
            for action in state.generate_possible_light_actions():
                next_state = state.apply_action(action)
                _, score = min_h(next_state, alpha, beta, depth - 1)
                if score > max_score:
                    best_action, max_score = action, score
                    alpha = max(alpha, score)
                if alpha >= beta:
                    break

            self.transposition_table[state_key] = (best_action, max_score)
            return best_action, max_score

        def min_h(state: GameState, alpha: Evaluation, beta: Evaluation, depth: int) -> tuple[Action, float]:
            """
            Minimizing player logic using Minimax or PVS during the endgame.
            
            Args:
                state (GameState): Current game state
                alpha (int): Alpha value
                beta (int): Beta value
                depth (int): Depth of the search tree
                
            Returns:
                tuple[Action, float]: The best action and its associated score
            """
            state_key = hash((hash(state), depth, False))
            if state_key in self.transposition_table:
                return self.transposition_table[state_key]

            if depth == 0 or state.is_done():
                return None, self.evaluate_action(state, scores_c)

            best_action, min_score = None, (float('inf'), float('inf'))
            for action in state.generate_possible_light_actions():
                next_state = state.apply_action(action)
                _, score = max_h(next_state, alpha, beta, depth - 1)
                if score < min_score:
                    best_action, min_score = action, score
                    beta = min(beta, score)
                if beta <= alpha:
                    break

            self.transposition_table[state_key] = (best_action, min_score)
            return best_action, min_score

        return max_h(current_state, (float('-inf'), float('-inf')), (float('inf'), float('inf')), depth)[0]
        
    def compute_action(self, current_state, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        compute the next action to play.
        
        Args:
            current_state (GameState): Current game state
            remaining_time (int, optional): Time limit for the action (default is 1e9)
            **kwargs: Additional keyword arguments
            
        Returns:
            Action: The selected action
        """
        game_phrase = self.get_game_phase(current_state)
        depth = self.dynamic_depth(game_phrase, remaining_time)

        print(f"Game Phase: {game_phrase}")
        print(f"Depth: {depth}")
        print(f"Remaining time: {remaining_time} seconds")

        return self.minmax_heuristic(current_state, remaining_time, depth, game_phrase)
