import backends
from aggro_treenode import TreeNode
import chess


weak_evaluator = backends.player_model_evaluator
strong_evaluator = backends.strong_evaluator


playing_as_white = None
opponent_probability_cutoff = 0.0
engine_culling_cutoff = 0.0
tendril_count = 16

curr_position = None

curr_eval = None


def get_best_move(position, nodes_limit, own_probability_cutoff, enemy_probability_cutoff):

    # Assign global variables from given parameters.
    global curr_position
    curr_position = position

    global playing_as_white
    playing_as_white = len(position) % 2 == 0

    global engine_culling_cutoff
    engine_culling_cutoff = own_probability_cutoff

    global opponent_probability_cutoff
    opponent_probability_cutoff = enemy_probability_cutoff

    # Get the current position's eval. We'll need it later.
    global curr_eval
    curr_eval = strong_evaluator.get_expected_outcome_from_moves(curr_position)
    curr_eval = (curr_eval + 1.0) / 2.0

    # Create tree root.
    root_node = TreeNode(total_value=0, visit_count=0, probability=1.0, position=position)

    # Perform a number of uct iterations equal to parameter nodes_limit.
    nodes_count = 0
    while nodes_count < nodes_limit // tendril_count:
        perform_iteration(root_node)
        nodes_count += 1

    # Return best move found so far.
    best_move = root_node.get_best_move()
    if best_move is None:
        best_move = get_best_legal_move(position)

    return best_move


def perform_iteration(root_node):

    #
    # TRAVERSAL
    #

    # Rustle up some leaf nodes to process.

    # TODO: Wait, hang on a minute: won't this just keep selecting the same leaf node?
    # No, wait, the hope here is that it will go down the tree randomly, since the select_child
    # function has randomness when dealing with a chance node.

    leaf_nodes = []
    for i in range(tendril_count):
        leaf_node = root_node
        while leaf_node.has_children():
            leaf_node = leaf_node.select_child(playing_as_white)

        leaf_nodes.append(leaf_node)

    leaf_nodes = remove_list_duplicates(leaf_nodes)

    #
    # EXPANSION
    #

    chance_nodes = []
    chance_node_positions = []

    for index in range(len(leaf_nodes)):
        # Skip expansion if the leaf node is terminal (i.e. checkmate or draw).
        if is_terminal_position(leaf_nodes[index].position):
            continue

        # Skip leaf node if it has not been visited.
        if leaf_nodes[index].visit_count <= 0:
            continue

        # Previously visited leaf nodes get expanded.

        if leaf_nodes[index].is_own_move(playing_as_white):
            # Leaf node is an OWN MOVE node.
            # So we create children for the node and pick one of those to consider.
            legal_moves = get_legal_moves_from_position(leaf_nodes[index].position)
            probabilities = []
            for i in range(len(legal_moves)):
                probabilities.append(1.0)
            probability_distribution = list(zip(legal_moves, probabilities))

            # This is kind of goofy, but it looks like for an OWN MOVE node, I just expand
            # the node with all legal moves assigned a probability of 1.0.
            # I guess this makes sense so you can just propagate value up the tree, multiplying
            # by probability each time. Since the engine can pick any move it wants,
            # a probability of 1.0 for each move makes sense.
            leaf_nodes[index].expand_with_probability_distribution(probability_distribution)
            leaf_nodes[index] = leaf_nodes[index].select_child(playing_as_white)

            # TODO: What the hell? Wouldn't we want to append this child node to the chance_nodes list?

        else:
            # Leaf node is a CHANCE node. We will need to expand it to consider it.
            chance_nodes.append(leaf_nodes[index])
            chance_node_positions.append(leaf_nodes[index].position)

    # Get probabilities from MAIA.
    chance_evaluations = weak_evaluator.get_evaluations_from_moves(chance_node_positions)

    # For each CHANCE node, expand it according to MAIA's policy
    for index in range(len(chance_nodes)):
        leaf_index = leaf_nodes.index(chance_nodes[index])
        probability_distribution = chance_evaluations[index].move_policy_list

        probability_distribution = get_simplified_probability_distribution(probability_distribution,
                                                                           opponent_probability_cutoff)

        leaf_nodes[leaf_index].expand_with_probability_distribution(probability_distribution)
        leaf_nodes[leaf_index] = leaf_nodes[leaf_index].select_child(playing_as_white)

    #
    # ROLLOUT
    #

    # Note: The "Rollout" phase gets its name from the typical Monte-Carlo method of randomly rolling out the rest
    # of the game randomly in order to get a value guess. In this algorithm, we can just use our strong evaluator.

    positions = []

    for leaf in leaf_nodes:
        positions.append(leaf.position)

    # Get policy values from STRONG LEELA or STOCKFISH
    expected_values = strong_evaluator.get_expected_outcomes_from_moves(positions)

    # Get evals of these positions.
    for index in range(len(leaf_nodes)):
        if is_terminal_position(leaf_nodes[index].position):
            # The * 10 here in the next line is just to really make the engine give a fuck about delivering mate.
            # Previously, it would be similarly jazzed about a winning position and mate.
            # Even though it's supposed to care about delivering mate quickly, a boring winning position in one move
            # is still evaluated better than a mate in two, for example.
            # Really, we should probably just make a function that maps the 0-1 eval to an exponential function.
            expected_values[index] = get_result(leaf_nodes[index].position, playing_as_white) * 10
        else:
            expected_values[index] = (expected_values[index] + 1.0) / 2.0

            if not playing_as_white:
                expected_values[index] = 1.0 - expected_values[index]  # Reverse objective value if playing as black

        # IMPORTANT: This is the aggro part.
        # After getting the raw expected value for the outcome, there's one more factor we care about.
        # We want to divide the eval increase by the number of moves it takes us to get there.
        moves_it_will_take = len(leaf_nodes[index].position) - len(curr_position)
        if moves_it_will_take == 0:
            continue
        expected_values[index] = (expected_values[index] - curr_eval) / (moves_it_will_take / 2.0)

    #
    # BACKPROPAGATION
    #

    # IMPORTANT: Here's the aggro part.
    # The expected values

    for index in range(len(leaf_nodes)):

        # For each leaf node, we add 1 to its visit count and add our eval to its total value.
        # Repeat for all parents of that leaf node.

        curr_node = leaf_nodes[index]

        while curr_node is not None:
            curr_node.visit_count += 1
            curr_node.total_value += expected_values[index]

            curr_node = curr_node.parent


def get_legal_moves_from_position(moves_list):

    if engine_culling_cutoff <= 0.0:
        board = chess.Board()
        for move in moves_list:
            board.push_uci(move)

        legal_moves = []
        for legal_move in board.legal_moves:
            legal_moves.append(legal_move.uci())

    else:
        evaluation = strong_evaluator.get_full_evaluation_from_moves(moves_list)
        policies_list = evaluation.move_policy_list

        # Remove the most unpromising moves.
        policies_list = get_simplified_probability_distribution(policies_list, engine_culling_cutoff)

        legal_moves = []
        for policy in policies_list:
            legal_moves.append(policy[0])

    return legal_moves


def is_terminal_position(position):
    board = chess.Board()
    for move in position:
        board.push_uci(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return True
    else:
        return False


def get_result(position, is_white):

    board = chess.Board()
    for move in position:
        board.push_uci(move)
    outcome = board.outcome(claim_draw=True)
    if outcome.result() == "1-0":
        result = 1.0
    elif outcome.result() == "0-1":
        result = 0.0
    else:
        result = 0.5

    if not is_white:
        result = 1.0 - result

    return result


def get_simplified_probability_distribution(distribution, cutoff):

    new_distribution = []
    new_distribution_sum = 0.0
    for element in distribution:
        if element[1] < cutoff:
            pass
        else:
            new_distribution_sum += element[1]

    for element in distribution:
        if element[1] < cutoff:
            pass
        else:
            new_distribution.append((element[0], element[1] / new_distribution_sum))

    return new_distribution


def probability_distribution_sort(entry):
    return entry[1]


# Quickly return the legal move with the highest policy value.
def get_best_legal_move(position):

    evaluation = strong_evaluator.get_full_evaluation_from_moves(position)
    legal_moves = evaluation.move_policy_list
    legal_moves.sort(key=probability_distribution_sort, reverse=True)

    return legal_moves[0][0]


def remove_list_duplicates(input_list):
    output_list = []
    for element in input_list:
        if element not in output_list:
            output_list.append(element)

    return output_list
