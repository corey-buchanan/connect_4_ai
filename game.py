import pygame
from time import sleep, time
from random import randrange

pygame.init()
screen = pygame.display.set_mode((560, 580))
clock = pygame.time.Clock()
running = True
player_turn = True
winner_declared = False
draw = False
last_time = time()

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

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP and player_turn == True:
            make_move(pygame.mouse.get_pos()[0] // 80)
            last_time = time()

    if time() - last_time > 0.25:
        while (not winner_declared and not player_turn):
            make_move(randrange(7))

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # RENDER YOUR GAME HERE

    render_dots()
    
    if (winner_declared or draw):
        render_winner()

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()