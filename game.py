from time import time
import pickle
import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Connect4Game:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.game_board = np.full((rows, columns), -1, dtype=int)
        self.player_turn = True
        self.winner_state = 0  # Initialize the winner state as "No winner yet"
        # Initialize other game-related attributes

    def make_move(self, column):
        if self.winner_state or not (0 <= column < self.columns) or self.is_column_full(column):
            return
        
        token = 1 if self.player_turn else 0

        row = self.find_next_empty_row(token, column)
        self.game_board[row, column] = token
        self.check_winner(row, column)

        self.player_turn = not self.player_turn

    def is_column_full(self, column):
        return self.game_board[self.rows - 1, column] != -1

    def find_next_empty_row(self, token, column):
        for row in range(self.rows):
            if self.game_board[row, column] == -1:
                return row

    def render(self, screen):
        self.render_tokens(screen)
        self.render_winner(screen)

    def render_tokens(self, screen):
        for row in range(self.rows):
            for col in range(self.columns):
                token = self.game_board[row, col]
                if token == -1:  # Empty cell
                    continue
                elif token == 0:  # Blue token
                    color = "blue"
                else:  # Red token
                    color = "red"
                
                # Calculate token position
                x = col * 80 + 40
                y = 540 - row * 80
                
                # Draw shadow slightly offset below the token
                self.draw_shadow(screen, x + 5, y + 5)  # Adjust the offset as needed
                
                # Draw the token
                pygame.draw.circle(screen, color, [x, y], 40)

    def draw_shadow(self, screen, x, y):
        shadow_color = (50, 50, 50)  # Define the shadow color
        pygame.draw.circle(screen, shadow_color, [x, y], 40)

    def render_winner(self, screen):
        font = pygame.font.SysFont(None, 64)
        if self.winner_state == 0:
            winner_text = f"{'Red' if self.player_turn else 'Blue'}'s turn"
        elif self.winner_state == 1:
            winner_text = "Draw!"
        elif self.winner_state == 2:
            winner_text = "User Wins!"
        elif self.winner_state == 3:
            winner_text = "AI Wins!"
        text_surface = font.render(winner_text, True, "black")
        screen.blit(text_surface, (170, 30))

    def process_game_state(self):
        # Create an empty 3D array to represent the processed game state
        processed_state = np.zeros((self.game_board.shape[0], self.game_board.shape[1], 3), dtype=int)

        # Fill the array with appropriate values based on the game board
        for row in range(self.game_board.shape[0]):
            for col in range(self.game_board.shape[1]):
                token = self.game_board[row, col]
                if token == -1:  # Empty cell
                    continue
                elif token == 0:  # Blue token
                    processed_state[row, col, 0] = 1
                else:  # Red token
                    processed_state[row, col, 1] = 1

        # Set the player's turn channel
        if self.player_turn:
            processed_state[:, :, 2] = 1

        return processed_state

    def check_winner(self, row, column):
        if self.check_horizontal_win(row, column) or \
           self.check_vertical_win(row, column) or \
           self.check_diagonal_axis1_win(row, column) or \
           self.check_diagonal_axis2_win(row, column):
            self.winner_state = 2 if self.player_turn else 3
        elif self.check_draw():
            self.winner_state = 1

    def check_horizontal_win(self, row, column):
        token = self.game_board[row, column]
        count = 1

        # Check to the left
        for c in range(column - 1, max(column - 4, -1), -1):
            if self.game_board[row, c] == token:
                count += 1
            else:
                break

        # Check to the right
        for c in range(column + 1, min(column + 4, self.columns)):
            if self.game_board[row, c] == token:
                count += 1
            else:
                break

        return count >= 4

    def check_vertical_win(self, row, column):
        token = self.game_board[row, column]
        count = 1

        # Check below
        for r in range(row - 1, max(row - 4, -1), -1):
            if self.game_board[r, column] == token:
                count += 1
            else:
                break

        return count >= 4

    def check_diagonal_axis1_win(self, row, column):
        token = self.game_board[row, column]
        count = 1

        # Check diagonally down-left
        for r, c in zip(range(row - 1, max(row - 4, -1), -1), range(column - 1, max(column - 4, -1), -1)):
            if self.game_board[r, c] == token:
                count += 1
            else:
                break

        # Check diagonally up-right
        for r, c in zip(range(row + 1, min(row + 4, self.rows)), range(column + 1, min(column + 4, self.columns))):
            if self.game_board[r, c] == token:
                count += 1
            else:
                break

        return count >= 4

    def check_diagonal_axis2_win(self, row, column):
        token = self.game_board[row, column]
        count = 1

        # Check diagonally down-right
        for r, c in zip(range(row - 1, max(row - 4, -1), -1), range(column + 1, min(column + 4, self.columns))):
            if self.game_board[r, c] == token:
                count += 1
            else:
                break

        # Check diagonally up-left
        for r, c in zip(range(row + 1, min(row + 4, self.rows)), range(column - 1, max(column - 4, -1), -1)):
            if self.game_board[r, c] == token:
                count += 1
            else:
                break

        return count >= 4

    def check_draw(self):
        return np.all(self.game_board != -1)


class Connect4AI:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        # Initialize AI-related attributes
        self.model = self.build_model()

    def build_model(self):
        input_shape = (self.rows, self.columns, 3)  # 2 channels: blue token, red token
        model = keras.Sequential([
            keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.columns, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_best_move(self, game, game_state):
        # Use the AI model to make a move
        input_state = np.expand_dims(game_state[:, :, :3], axis=0)
        predicted_probabilities = self.model.predict(input_state)[0]
        valid_moves = np.array([not game.is_column_full(col) for col in range(self.columns)])
        masked_probabilities = predicted_probabilities * valid_moves
        best_move = np.argmax(masked_probabilities)
        return best_move

    def train(self, dataset, num_epochs, batch_size):
        x_train = np.array(dataset['game_states'])
        y_train = np.array(dataset['rewards'])

        # Train the AI model using the dataset
        self.model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        try:
            self.model = keras.models.load_model(filename)
            print("Model loaded successfully.")
        except (OSError, ValueError):
            print("Model file not found or unable to load. Using a newly built model.")

# Save dataset to a file
def save_dataset(dataset, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        print("Dataset saved successfully.")
    except Exception as e:
        print("Failed to save dataset:", str(e))

# Load dataset from a file
def load_dataset(filename):
    try:
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
            print("Dataset loaded successfully.")
            return dataset
    except FileNotFoundError:
        print("Dataset file not found. Starting with an empty dataset.")
        return {
            'game_states': [],
            'moves': [],
            'rewards': []
        }
    except Exception as e:
        print("Failed to load dataset:", str(e))
        return {
            'game_states': [],
            'moves': [],
            'rewards': []
        }

def quit(ai):
    pygame.quit()
    exit()

# Initialize pygame and other constants
pygame.init()
screen = pygame.display.set_mode((560, 580))
clock = pygame.time.Clock()
rows = 6
columns = 7
running = True
num_epochs = 20
batch_size = 32
filename = 'connect4_data.dat'

# AI instances
ai = Connect4AI(rows, columns)

dataset = load_dataset(filename)

ai.load_model("model.h5")  # Load the model from a file 

# Game loop
while True:
    # Start a new game
    game = Connect4Game(rows, columns)
    game_states = []
    moves = []

    while not game.winner_state:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit(ai)
            if event.type == pygame.MOUSEBUTTONUP and game.player_turn:
                game_state = game.process_game_state()
                user_move = (pygame.mouse.get_pos()[0] // 80)
                game.make_move(user_move)
                game_states.append(game_state)
                moves.append(user_move)
                last_time = time()

        if not game.player_turn:  # AI's turn
            game_state = game.process_game_state()
            ai_move = ai.get_best_move(game, game_state)  # Get user's move
            game.make_move(ai_move)
            game_states.append(game_state)
            moves.append(ai_move)
        
        screen.fill("white")  # Clear the screen
        game.render(screen)   # Render the game state
        pygame.display.flip()

    # game_state[:,:,2] is player_turn for that state
    if game.winner_state == 2: # User wins
        rewards = [0 if game_state[0,0,2] else -1 for game_state in game_states] # Assign negative reward for AI's moves
    elif game.winner_state == 3: # AI wins
        rewards = [0 if game_state[0,0,2] else 1 for game_state in game_states] # Assign positive reward for AI's moves
    else: # Draw
        rewards = [0] * len(moves)  # Assign neutral reward for all moves

    reshaped_rewards = np.repeat(np.array(rewards)[:, np.newaxis], 7, axis=1)

    dataset['game_states'].extend(game_states)
    dataset['moves'].extend(moves)

    for reward in reshaped_rewards:
        dataset['rewards'].append(reward)
    
    save_dataset(dataset, filename)

    ai.train(dataset, num_epochs, batch_size)
    ai.save_model("model.h5")
