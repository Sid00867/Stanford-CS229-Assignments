import numpy as np

class TicTacToe:
    """
    A model for the game of Tic-Tac-Toe.
    States are represented as 3x3 numpy arrays with values:
        0: empty
        1: player X
       -1: player O
    Actions are (row, col) tuples.
    """

    def __init__(self):
        self.state_shape = (3, 3)
        self.action_space = [(i, j) for i in range(3) for j in range(3)]

    def get_initial_state(self):
        """Return the initial empty board state."""
        return np.zeros(self.state_shape, dtype=int)

    def get_possible_actions(self, state):
        """Return a list of possible actions (empty cells) for the given state."""
        if self.is_terminal(state):
            return []
        return [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]

    def is_terminal(self, state):
        """Return True if the state is terminal (win or draw), else False."""
        winner = self.get_winner(state)
        # A state is terminal if a player has won or if no moves are left.
        return winner is not None

    def get_winner(self, state):
        """
        Return 1 if X wins, -1 if O wins, 0 if draw, None if not terminal.
        """
        for player in [1, -1]:
            # Check rows, columns, and diagonals
            if any(np.all(state[i, :] == player) for i in range(3)) or \
               any(np.all(state[:, j] == player) for j in range(3)) or \
               np.all(np.diag(state) == player) or \
               np.all(np.diag(np.fliplr(state)) == player):
                return player
        
        # Check for a draw (no empty cells left)
        if np.all(state != 0):
            return 0
            
        # If the game is not over, return None
        return None

    # CHANGE: Simplified the reward function.
    # The value function will now be from the perspective of Player X.
    # So, the reward is 1 if X wins, -1 if O wins, and 0 otherwise.
    def state_reward(self, state):
        """
        Returns the reward for a terminal state from the perspective of Player X.
        1 if X wins, -1 if O wins, 0 for a draw.
        """
        winner = self.get_winner(state)
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        else:
            return 0

    def step(self, state, action, player):
        """
        Take action (row, col) as 'player' in the given state.
        Returns the next_state.
        """
        if state[action] != 0:
            raise ValueError("Invalid action: cell already occupied.")
        next_state = state.copy()
        next_state[action] = player
        # The reward and done status are handled by the value iteration loop now.
        return next_state

    def get_board_number(self, state):
        """Map a 3x3 board state to a unique number (base-3 encoding)."""
        flat = state.flatten()
        base3_digits = np.where(flat == 0, 0, np.where(flat == 1, 1, 2))
        board_number = 0
        for d in base3_digits:
            board_number = board_number * 3 + d
        return board_number

    def board_number_to_state(self, board_number):
        """Reverse of get_board_number."""
        board_number = min(board_number, 19681)
        base3_digits = []
        n = board_number
        for _ in range(9):
            base3_digits.append(n % 3)
            n //= 3
        base3_digits = base3_digits[::-1]
        flat = np.array([0 if d == 0 else (1 if d == 1 else -1) for d in base3_digits])
        return flat.reshape((3, 3))

    def is_legal_configuration(self, state):
        """Check if a given board configuration is legal."""
        num_x = np.sum(state == 1)
        num_o = np.sum(state == -1)

        if not (num_x == num_o or num_x == num_o + 1):
            return False

        x_wins = self.get_winner(state) == 1
        o_wins = self.get_winner(state) == -1

        if x_wins and o_wins:
            return False
        if x_wins and num_x != num_o + 1:
            return False
        if o_wins and num_x != num_o:
            return False

        return True

    def get_current_player(self, state):
        """Determines whose turn it is from the given state."""
        num_x = np.sum(state == 1)
        num_o = np.sum(state == -1)
        if self.is_terminal(state):
            return 0 # Game is over
        return 1 if num_x == num_o else -1

    def render(self, state):
        """Print the board."""
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print()
        for i in range(3):
            row = [symbols[state[i, j]] for j in range(3)]
            print(' | '.join(row))
            if i < 2:
                print('--+---+--')
        print()


class MDP:
    def __init__(self) -> None:
        self.game = TicTacToe()

    def initialize_mdp_data(self):
        """Initializes the value table."""
        # The value table represents the "value" of each state from X's perspective.
        value = np.zeros(19682) 
        return {'value': value}

    # CHANGE: Implemented Minimax logic in choose_action.
    # The agent now considers the opponent's goal.
    def choose_action(self, state, mdp_data, player):
        """
        Chooses the best action based on the minimax principle.
        - If player is X (1), it chooses the move with the maximum future value.
        - If player is O (-1), it chooses the move with the minimum future value.
        """
        value = mdp_data['value']
        valid_actions = self.game.get_possible_actions(state)
        if not valid_actions:
            return None

        action_values = []
        for action in valid_actions:
            next_state = self.game.step(state, action, player)
            board_num = self.game.get_board_number(next_state)
            action_values.append(value[board_num])

        if player == 1:  # Player X wants to MAXIMIZE the value
            best_value = np.max(action_values)
            best_indices = [i for i, v in enumerate(action_values) if v == best_value]
        else:  # Player O wants to MINIMIZE the value (from X's perspective)
            best_value = np.min(action_values)
            best_indices = [i for i, v in enumerate(action_values) if v == best_value]
        
        chosen_idx = np.random.choice(best_indices)
        return valid_actions[chosen_idx]

    # CHANGE: This is the core of the new algorithm.
    # The function now updates a single value table for all states,
    # applying the minimax principle.
    def update_mdp_value(self, mdp_data, tolerance, gamma):
        """
        Performs value iteration on the Tic-Tac-Toe MDP using Minimax.
        This function now trains a single, unified agent.
        """
        num_states = 19682
        iters = 0
        value = mdp_data['value']

        # --- Pre-computation Step ---
        updatable_states = []
        transitions = {}

        for state_num in range(num_states):
            state_arr = self.game.board_number_to_state(state_num)
            
            # CHANGE: We now consider ALL legal states, not just for one player.
            if self.game.is_legal_configuration(state_arr):
                # CHANGE: Set the value for terminal states directly. This is the starting point for backpropagation.
                if self.game.is_terminal(state_arr):
                    value[state_num] = self.game.state_reward(state_arr)
                else:
                    # If not terminal, it's a state we need to update.
                    updatable_states.append(state_num)
                    
                    # CHANGE: Pre-compute transitions based on the actual current player of the state.
                    current_player = self.game.get_current_player(state_arr)
                    next_state_nums = []
                    for action in self.game.get_possible_actions(state_arr):
                        next_state = self.game.step(state_arr, action, current_player)
                        board_num = self.game.get_board_number(next_state)
                        next_state_nums.append(board_num)
                    
                    if next_state_nums:
                        transitions[state_num] = np.array(next_state_nums, dtype=int)

        # --- Minimax Value Iteration Loop ---
        while True:
            iters += 1
            value_old = value.copy()

            for state in updatable_states:
                state_arr = self.game.board_number_to_state(state)
                current_player = self.game.get_current_player(state_arr)
                next_state_indices = transitions.get(state)

                if next_state_indices is not None and next_state_indices.size > 0:
                    next_values = value_old[next_state_indices]
                    
                    # CHANGE: Apply the Minimax update rule.
                    # If it's X's turn, he will maximize his outcome.
                    # If it's O's turn, he will minimize X's outcome.
                    if current_player == 1:
                        best_next_value = np.max(next_values)
                    else: # current_player == -1
                        best_next_value = np.min(next_values)
                    
                    # The Bellman update, based on the best outcome from the next state.
                    value[state] = gamma * best_next_value
                
            if np.max(np.abs(value - value_old)) < tolerance:
                break
        
        print(f"Converged in {iters} iterations.")


# CHANGE: The main function is now simplified to train a single, unified agent.
def main(gamma=0.9, tolerance=1e-4):
    """Initializes and trains a single MDP agent."""
    np.random.seed(42)
    mdp_agent = MDP()
    mdp_data = mdp_agent.initialize_mdp_data()

    print("Training agent with Minimax Value Iteration...")
    # CHANGE: Call the updated function once to train the unified value table.
    mdp_agent.update_mdp_value(mdp_data, tolerance, gamma)

    print("Training complete.")
    return mdp_agent, mdp_data

# CHANGE: Evaluation now uses the single trained agent for both players.
def evaluate_agent(mdp_agent, mdp_data, num_games=1000):
    """Evaluates the trained agent by making it play against itself."""
    results = {1: 0, -1: 0, 0: 0}  # X wins, O wins, Draws
    for _ in range(num_games):
        state = mdp_agent.game.get_initial_state()
        done = False
        while not done:
            player = mdp_agent.game.get_current_player(state)
            if mdp_agent.game.is_terminal(state):
                done = True
                continue

            # Both 'X' and 'O' use the same minimax logic from choose_action.
            action = mdp_agent.choose_action(state, mdp_data, player)
            if action is None:
                break
            state = mdp_agent.game.step(state, action, player)

        winner = mdp_agent.game.get_winner(state)
        if winner is not None:
            results[winner] += 1
    print(f"\nSelf-Play Evaluation ({num_games} games):")
    print(f"X wins: {results[1]}, O wins: {results[-1]}, Draws: {results[0]}")

# CHANGE: The game loop is simplified to use the single agent.
def play_game_with_user(mdp_agent, mdp_data):
    """Manages the interactive game session between a user and the agent."""
    game = mdp_agent.game
    print("\nLet's play a game! You can play as X (1) or O (-1).")
    
    while True:
        try:
            user_player = int(input("Enter 1 to play as X, -1 to play as O: "))
            if user_player in [1, -1]:
                break
            else:
                print("Please enter 1 or -1.")
        except ValueError:
            print("Invalid input. Please enter 1 or -1.")

    state = game.get_initial_state()
    done = False
    
    while not done:
        game.render(state)
        player = game.get_current_player(state)

        if player == user_player:
            # User's turn
            valid = False
            while not valid:
                try:
                    move = int(input("Enter your move (0-8, left to right, top to bottom): "))
                    action = (move // 3, move % 3)
                    if 0 <= move <= 8 and state[action] == 0:
                        valid = True
                    else:
                        print("Invalid move. Try again.")
                except (ValueError, IndexError):
                    print("Invalid input. Enter a number between 0 and 8.")
        else:
            # Agent's turn
            print("Agent is thinking...")
            action = mdp_agent.choose_action(state, mdp_data, player)
            print(f"Agent ({'X' if player == 1 else 'O'}) chooses move: {action[0]*3 + action[1]}")

        state = game.step(state, action, player)
        done = game.is_terminal(state)
        
    game.render(state)
    winner = game.get_winner(state)
    if winner == user_player:
        print("Congratulations! You win!")
    elif winner == 0:
        print("It's a draw!")
    else:
        print("The agent wins! Better luck next time.")


if __name__ == '__main__':
    agent, data = main()
    evaluate_agent(agent, data)
    
    while True:
        play_game_with_user(agent, data)
        play_again = input("Do you want to play another game? (y/n): ").strip().lower()
        if play_again not in ['y', 'yes']:
            print("Thanks for playing!")
            break