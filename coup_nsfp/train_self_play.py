from agent import NFSPAgent
from game import CoupGame
import numpy as np
from config import ACTIONS, NUM_CARD_TYPES, MAX_TURNS_PER_GAME


def get_state_for_player(game, player, p0_hist, p1_hist):
    game_state = game.get_state()
    opponent = 1 - player
    
    my_cards = np.array(game_state['roles'][player], dtype=np.float32)
    my_coins = float(game_state['players_coins'][player]) / 10.0
    opp_coins = float(game_state['players_coins'][opponent]) / 10.0
    my_inf = float(game_state['influence'][player])
    opp_inf = float(game_state['influence'][opponent])
    
    if player == 0:
        my_hist = p0_hist.astype(np.float32)
        opp_hist = p1_hist.astype(np.float32)
    else:
        my_hist = p1_hist.astype(np.float32)
        opp_hist = p0_hist.astype(np.float32)
    
    discard = np.array(game_state['discard_pile'], dtype=np.float32)
    deck_size = float(game_state['deck_size']) / 15.0
    
    state = np.concatenate([
        my_cards,
        np.array([my_coins], dtype=np.float32),
        np.array([opp_coins], dtype=np.float32),
        np.array([my_inf], dtype=np.float32),
        np.array([opp_inf], dtype=np.float32),
        my_hist,
        opp_hist,
        discard,
        np.array([deck_size], dtype=np.float32),
    ])
    
    return state


def update_action_history(p0_hist, p1_hist, player, action):
    action_name = ACTIONS[action]
    
    if action_name == "TAX":
        card_claimed = 0
    elif action_name == "ASSASSINATE":
        card_claimed = 1
    elif action_name == "EXCHANGE":
        card_claimed = 2
    elif action_name == "STEAL":
        card_claimed = 3
    else:
        return
    
    if player == 0:
        p0_hist[card_claimed] = min(2, p0_hist[card_claimed] + 1)
    else:
        p1_hist[card_claimed] = min(2, p1_hist[card_claimed] + 1)


def calculate_reward(game, player, pre_coins, pre_opp_coins, pre_inf, pre_opp_inf):
    game_state = game.get_state()
    post_coins = game_state['players_coins'][player]
    post_opp_coins = game_state['players_coins'][1 - player]
    post_inf = game_state['influence'][player]
    post_opp_inf = game_state['influence'][1 - player]
    
    reward = 0.0
    
    if game.is_game_over():
        winner = game.get_winner()
        if winner == player:
            reward += 1.0
        else:
            reward -= 1.0
    
    return reward


def self_play_training(num_episodes=10_000, save_prefix="nfsp_coup"):
    action_to_move = {
        0: 4,   
        1: 5,
        2: 0,
        3: 3,
        4: 2,
        5: 1,
        6: 6,
    }
    
    # Inverse mapping: game move -> agent action
    move_to_action = {v: k for k, v in action_to_move.items()}
    
    dummy_game = CoupGame()
    p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    state_dim = get_state_for_player(dummy_game, 0, p0_hist, p1_hist).shape[0]

    agent1 = NFSPAgent(state_dim)
    agent2 = NFSPAgent(state_dim)

    wins_agent1 = 0
    wins_agent2 = 0

    for ep in range(1, num_episodes + 1):
        game = CoupGame()
        p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        turn = 0
        
        def block_callback(player, block_type):
            agent = agent1 if player == 0 else agent2
            current_state = get_state_for_player(game, player, p0_hist, p1_hist)
            block_action, state_with_type = agent.select_block_action(current_state, block_type, training=True)
            agent.store_block_transition(state_with_type, block_action)
            agent.train_block()
            return block_action == 1
        
        def challenge_callback(player, claimed_card):
            agent = agent1 if player == 0 else agent2
            current_state = get_state_for_player(game, player, p0_hist, p1_hist)
            if isinstance(claimed_card, (list, tuple)):
                card_val = claimed_card[0]
            else:
                card_val = claimed_card
            challenge_action, state_with_card = agent.select_challenge_action(current_state, card_val, training=True)
            agent.store_challenge_transition(state_with_card, challenge_action)
            agent.train_challenge()
            return challenge_action == 1
        
        game.block_callback = block_callback
        game.challenge_callback = challenge_callback
        
        state = get_state_for_player(game, game.current_player, p0_hist, p1_hist)
        done = False

        while not done:
            current_player = game.current_player
            agent = agent1 if current_player == 0 else agent2
            
            # Get valid moves from game and convert to agent action space
            valid_game_moves = game.get_valid_moves(current_player)
            valid_actions = [move_to_action[move] for move in valid_game_moves]
            
            action, mode_used = agent.select_action(state, training=True, valid_actions=valid_actions)
            
            game_state_pre = game.get_state()
            pre_coins = game_state_pre['players_coins'][current_player]
            pre_opp_coins = game_state_pre['players_coins'][1 - current_player]
            pre_inf = game_state_pre['influence'][current_player]
            pre_opp_inf = game_state_pre['influence'][1 - current_player]
            
            game_move = action_to_move[action]
            update_action_history(p0_hist, p1_hist, current_player, action)
            game.play_move(current_player, game_move)
            
            reward = calculate_reward(game, current_player, pre_coins, pre_opp_coins, pre_inf, pre_opp_inf)
            
            done = game.is_game_over()
            turn += 1
            if turn >= MAX_TURNS_PER_GAME:
                done = True
            
            # get next state from same player's perspective before switching
            next_state = get_state_for_player(game, current_player, p0_hist, p1_hist)
            
            agent.store_transition(state, action, reward, next_state, done, mode_used)
            agent.train_step()
            
            # switch turns
            game.current_player = (game.current_player + 1) % 2
            if not done:
                state = get_state_for_player(game, game.current_player, p0_hist, p1_hist)

        game_state_final = game.get_state()
        if game_state_final['influence'][0] > game_state_final['influence'][1]:
            wins_agent1 += 1
        elif game_state_final['influence'][1] > game_state_final['influence'][0]:
            wins_agent2 += 1

        if ep % 100 == 0:
            print(f"Episode {ep}/{num_episodes} | Wins A1: {wins_agent1} | Wins A2: {wins_agent2}")

    # Save the agent that won more games
    if wins_agent1 >= wins_agent2:
        agent1.save(save_prefix)
        print(f"\nSaving Agent 1 (won {wins_agent1} games vs {wins_agent2})")
    else:
        agent2.save(save_prefix)
        print(f"\nSaving Agent 2 (won {wins_agent2} games vs {wins_agent1})")


if __name__ == "__main__":
    # Adjust num_episodes as needed
    self_play_training(num_episodes=5000, save_prefix="nfsp_coup_demo")
