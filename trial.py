import config
import player

# Performs a number of trials, comparing Polecat against Stockfish.
# Outputs average number of half-moves per game for each engine.

trials = config.NUMBER_OF_TRIALS

total_polecat_half_moves = 0
total_sf_half_moves = 0

print("Running trials...")

player.simulate_player = True
player.suppress_game_text = True

player.computer_engine = "aggro fixed stochastic uct"

progress_report_interval = 2

for i in range(trials):
    if i > 0 and i % progress_report_interval == 0:
        print("Polecat trials completion " + str(i / trials * 100) + "%")
        print("Current Polecat average half-moves: " + str(total_polecat_half_moves / (i + 1)))
    played_game = player.play_game(play_random=False, is_computer_white=(i % 2 == 0))
    total_polecat_half_moves += len(played_game)

print("Polecat trial finished.")
print(str(trials) + " trials with average half-move count of " + str(total_polecat_half_moves / trials))

player.computer_engine = "stockfish"

for i in range(trials):
    if i > 0 and i % progress_report_interval == 0:
        print("Stockfish trials completion " + str(i / trials * 100) + "%")
        print("Current Stockfish average half-moves: " + str(total_sf_half_moves / (i + 1)))
    played_game = player.play_game(play_random=False, is_computer_white=(i % 2 == 0))
    total_sf_half_moves += len(played_game)


print("Completed " + str(trials) + " trials.")
print("")
print("  Polecat average half-moves: " + str(total_polecat_half_moves / trials))
print("Stockfish average half-moves: " + str(total_sf_half_moves / trials))
