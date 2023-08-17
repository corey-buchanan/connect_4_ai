import pygame
from time import sleep, time
from random import randrange
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

pygame.init()
screen = pygame.display.set_mode((560, 580))
clock = pygame.time.Clock()
last_time = time()
rows = 6
columns = 7
num_epochs = 20
batch_size = 32

def render_winner():
    font = pygame.font.SysFont(None, 64)
    winner_text = ("Red" if player_turn else "Blue") + " Wins!"
    text_surface = font.render("Draw!" if draw else winner_text, True, "black")
    screen.blit(text_surface, (170, 30))

def exists_and_filled(board_pos):
    return (0 <= board_pos[0] < 7 and 0 <= board_pos[1] < len(game_board[board_pos[0]]) and game_board[board_pos[0]][board_pos[1]] == player_turn)

def filled_in_direction(board_pos, direction):
    direction_count = 0
    while (exists_and_filled((board_pos[0] + direction[0], board_pos[1] + direction[1]))):
        direction_count += 1
        board_pos = (board_pos[0] + direction[0], board_pos[1] + direction[1])
    return direction_count


def check_horizontal(board_pos):
    horizontal_run = filled_in_direction(board_pos, (-1, 0)) + filled_in_direction(board_pos, (1, 0)) + 1
    # print("Horizontal run: ", horizontal_run)
    if (horizontal_run >= 4):
        return True
    return False

def check_vertical(board_pos):
    vertical_run = filled_in_direction(board_pos, (0, -1)) + filled_in_direction(board_pos, (0, 1)) + 1
    # print("Vertical run: ", vertical_run)
    if (vertical_run >= 4):
        return True
    return False

def check_diagonal(board_pos):
    diagonal_run_1 = filled_in_direction(board_pos, (-1, -1)) + filled_in_direction(board_pos, (1, 1)) + 1
    diagonal_run_2 = filled_in_direction(board_pos, (-1, 1)) + filled_in_direction(board_pos, (1, -1)) + 1
    # print("Vertical run 1: ", diagonal_run_1)
    # print("Vertical run 2: ", diagonal_run_2)
    if (diagonal_run_1 >= 4):
        return True
    if (diagonal_run_2 >= 4):
        return True
    return False

def check_winner(board_pos):
    global winner_declared
    if (check_horizontal(board_pos) or check_vertical(board_pos) or check_diagonal(board_pos)):
        winner_declared = True
        return True
    return False

def check_draw():
    global draw
    columns_filled = 0
    for column in game_board:
        if len(column) == 6:
            columns_filled += 1

    if columns_filled == 7:
        print("Draw!")
        draw = True


def make_move(column_number):
    global player_turn, winner_declared
    column = game_board[column_number]

    if len(column) > 5 or winner_declared or draw:
        return

    column.append(1 if player_turn else 0)

    if check_winner((column_number, len(column) - 1)):
        return

    player_turn = not player_turn
    check_draw()

def render_dots():
    col_pos = 40
    for col in game_board:
        row_pos = 540
        for row in col:
            pygame.draw.circle(screen, "red" if row == 1 else "blue", [col_pos, row_pos], 40)
            row_pos -= 80
        col_pos += 80

# Initialize a blank dataset
dataset = {
    'game_states': [],
    'moves': []
}

# Create the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(rows, columns, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(columns, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(x=dataset['game_states'], y=dataset['moves'], epochs=num_epochs, batch_size=batch_size)

def process_game_state(game_board, player_turn):
    processed_board = np.zeros((6, 7, 2), dtype=int)  # Initialize an empty 3D array
    
    for col_idx, column in enumerate(game_board):
        for row_idx, cell in enumerate(column):
            if cell == 0:  # Blue
                processed_board[row_idx, col_idx, 0] = 1
            elif cell == 1:  # Red
                processed_board[row_idx, col_idx, 1] = 1
    
    # Mark the current player's turn
    if player_turn:
        processed_board[:, :, 1] = 1  # Set the second channel to 1 for the current player
    
    return processed_board

def get_valid_moves(game_board):
    valid_moves = np.zeros(7, dtype=int)  # Initialize an array to store valid moves
    
    for col_idx, column in enumerate(game_board):
        if len(column) < 6:  # Check if the column isn't completely filled
            valid_moves[col_idx] = 1
    
    return valid_moves

# During gameplay, use the model to make a move
def get_best_move(game_state):
    input_state = np.expand_dims(game_state, axis=0)  # Add batch dimension
    predicted_probabilities = model.predict(input_state)[0]
    valid_moves = get_valid_moves(game_board)
    best_move = np.argmax(predicted_probabilities * valid_moves)
    return best_move

# Self-play loop
num_iterations = 5  # Adjust as needed
for iteration in range(num_iterations):
    game_states = []
    moves = []
    
    # Play a game
    # Initialize the game board and other variables

    # Game board - 7 columns, 6 rows
    game_board = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]

    running = True
    player_turn = True
    winner_declared = False
    draw = False
    
    while not (winner_declared or draw):
        if player_turn: # AI agent's turn
            game_state = process_game_state(game_board, player_turn)
            ai_move = get_best_move(game_state)
            make_move(ai_move)
            game_states.append(game_state)
            moves.append(ai_move)
        else: # Opponent's (AI agent's) turn
            game_state = process_game_state(game_board, player_turn)
            opponent_move = get_best_move(game_state)
            make_move(opponent_move)
            game_states.append(game_state)
            moves.append(opponent_move)

        screen.fill("white")
        render_dots()
        pygame.display.flip()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Pause briefly to allow the game state to be seen
        pygame.time.delay(100) # Delay for 250 milliseconds
        
        last_time = time()
        while (time() - last_time < 0.1):
            # Continue the self-play loop without rendering, to keep the game logic running
            pass
    
    # Store data from the game in the dataset
    dataset['game_states'].extend(game_states)
    dataset['moves'].extend(moves)

# Train the model using the collected dataset
x_train = np.array(dataset['game_states'])
# Convert moves to one-hot encoded vectors
y_train = to_categorical(dataset['moves'], num_classes=columns)

# Train the model
model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size)

# Game board - 7 columns, 6 rows
game_board = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
]

running = True
player_turn = True
winner_declared = False
draw = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP and player_turn:
            make_move(pygame.mouse.get_pos()[0] // 80)
            last_time = time()

    if time() - last_time > 0.25 and not winner_declared and not player_turn:
        # Use the AI model to make a move (replace this with your actual usage)
        ai_move = get_best_move(process_game_state(game_board, player_turn))
        make_move(ai_move)

    screen.fill("white")
    render_dots()

    if winner_declared or draw:
        render_winner()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()