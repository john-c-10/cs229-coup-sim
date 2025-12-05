from agent import NFSPAgent
from game import CoupGame
import numpy as np
from config import ACTIONS, NUM_CARD_TYPES, MAX_TURNS_PER_GAME
from baselineTraining import NoLieBot


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


def update_action_history(p0_hist, p1_hist, player, game_move):
    # Map game moves to claimed cards:
    # 0: Tax (claims Duke)
    # 1: Assassinate (claims Assassin)
    # 2: Exchange (claims Ambassador)
    # 3: Steal (claims Captain)
    # 4: Income (no claim)
    # 5: Foreign Aid (no claim)
    # 6: Coup (no claim)
    
    card_claimed = None
    if game_move == 0:  # Tax
        card_claimed = 0  # Duke
    elif game_move == 1:  # Assassinate
        card_claimed = 1  # Assassin
    elif game_move == 2:  # Exchange
        card_claimed = 2  # Ambassador
    elif game_move == 3:  # Steal
        card_claimed = 3  # Captain
    else:
        return  # No card claimed for Income, Foreign Aid, or Coup
    
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
    
    # Terminal reward (most important)
    if game.is_game_over():
        winner = game.get_winner()
        if winner == player:
            reward += 10.0  # Win bonus
        else:
            reward -= 10.0  # Loss penalty
        return reward
    
    # Intermediate rewards for shaping
    # Reward gaining coins (but cap it to avoid exploitation)
    coin_gain = (post_coins - pre_coins) * 0.1
    reward += coin_gain
    
    # Reward opponent losing coins
    opp_coin_loss = (pre_opp_coins - post_opp_coins) * 0.1
    reward += opp_coin_loss
    
    # Heavy penalty for losing influence
    inf_loss = (pre_inf - post_inf) * 3.0
    reward -= inf_loss
    
    # Heavy reward for opponent losing influence
    opp_inf_loss = (pre_opp_inf - post_opp_inf) * 3.0
    reward += opp_inf_loss
    
    return reward


def pretrain_against_baseline(agent, num_episodes=500):
    # very limited pretrain
    print(f"Pretraining agent against NoLieBot for {num_episodes} rounds")
    
    baseline = NoLieBot()
    
    action_to_move = {
        0: 4, 1: 5, 2: 0, 3: 3, 4: 2, 5: 1, 6: 6,
    }
    move_to_action = {v: k for k, v in action_to_move.items()}
    
    wins = 0
    losses = 0
    
    for ep in range(1, num_episodes + 1):
        game = CoupGame()
        p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        turn = 0
        
        agent_player = ep % 2
        baseline_player = 1 - agent_player
        
        def block_callback(player, block_type):
            if player == agent_player:
                current_state = get_state_for_player(game, player, p0_hist, p1_hist)
                block_action, state_with_type = agent.select_block_action(current_state, block_type, training=True)
                agent.store_block_transition(state_with_type, block_action)
                return block_action == 1
            else:
                return baseline.decide_block(game, player, block_type)
        
        def challenge_callback(player, claimed_card):
            if player == agent_player:
                current_state = get_state_for_player(game, player, p0_hist, p1_hist)
                if isinstance(claimed_card, (list, tuple)):
                    card_val = claimed_card[0]
                else:
                    card_val = claimed_card
                challenge_action, state_with_card = agent.select_challenge_action(current_state, card_val, training=True)
                agent.store_challenge_transition(state_with_card, challenge_action)
                return challenge_action == 1
            else:
                return baseline.decide_challenge(game, player, claimed_card)
        
        game.block_callback = block_callback
        game.challenge_callback = challenge_callback
        
        done = False
        
        while not done:
            current_player = game.current_player
            
            if current_player == agent_player:
                state = get_state_for_player(game, current_player, p0_hist, p1_hist)
                
                valid_game_moves = game.get_valid_moves(current_player)
                valid_actions = [move_to_action[move] for move in valid_game_moves]
                
                action, mode_used = agent.select_action(state, training=True, valid_actions=valid_actions)
                
                game_state_pre = game.get_state()
                pre_coins = game_state_pre['players_coins'][current_player]
                pre_opp_coins = game_state_pre['players_coins'][1 - current_player]
                pre_inf = game_state_pre['influence'][current_player]
                pre_opp_inf = game_state_pre['influence'][1 - current_player]
                
                game_move = action_to_move[action]
                update_action_history(p0_hist, p1_hist, current_player, game_move)
                game.play_move(current_player, game_move)
                
                reward = calculate_reward(game, current_player, pre_coins, pre_opp_coins, pre_inf, pre_opp_inf)
                
                done = game.is_game_over()
                turn += 1
                if turn >= MAX_TURNS_PER_GAME:
                    done = True
                
                next_state = get_state_for_player(game, current_player, p0_hist, p1_hist)
                
                agent.store_transition(state, action, reward, next_state, done, mode_used)
                agent.train_step()
            else:
                baseline_action = baseline.select_action(game, current_player)
                game.play_move(current_player, baseline_action)
                
                done = game.is_game_over()
                turn += 1
                if turn >= MAX_TURNS_PER_GAME:
                    done = True
            
            if not done:
                game.current_player = (game.current_player + 1) % 2
        
        agent.train_block()
        agent.train_challenge()
        
        game_state_final = game.get_state()
        if game_state_final['influence'][agent_player] > game_state_final['influence'][baseline_player]:
            wins += 1
        elif game_state_final['influence'][baseline_player] > game_state_final['influence'][agent_player]:
            losses += 1
        
        if ep % 100 == 0:
            win_rate = wins / 100
            print(f"Pretrain round {ep}/{num_episodes} | Win rate: {win_rate:.2%}")
            wins = 0
            losses = 0
    
    print("Pretraining complete")


def self_play_training(num_episodes=10000, save_prefix="nfsp_coup", pretrain=True):
    # Agent action to game move mapping
    action_to_move = {
        0: 4,   # Income
        1: 5,   # Foreign Aid
        2: 0,   # Tax
        3: 3,   # Steal
        4: 2,   # Exchange
        5: 1,   # Assassinate
        6: 6,   # Coup
    }
    
    # game move to agent action mapping
    move_to_action = {v: k for k, v in action_to_move.items()}
    
    dummy_game = CoupGame()
    p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    state_dim = get_state_for_player(dummy_game, 0, p0_hist, p1_hist).shape[0]

    agent1 = NFSPAgent(state_dim)
    agent2 = NFSPAgent(state_dim)
    
    if pretrain:
        pretrain_against_baseline(agent1, num_episodes=1000)
        pretrain_against_baseline(agent2, num_episodes=1000)

    wins_agent1 = 0
    wins_agent2 = 0

    for ep in range(1, num_episodes + 1):
        game = CoupGame()
        p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        turn = 0
        
        # store block/challenge transitions and train in batches
        def block_callback(player, block_type):
            agent = agent1 if player == 0 else agent2
            current_state = get_state_for_player(game, player, p0_hist, p1_hist)
            block_action, state_with_type = agent.select_block_action(current_state, block_type, training=True)
            agent.store_block_transition(state_with_type, block_action)
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
            return challenge_action == 1
        
        game.block_callback = block_callback
        game.challenge_callback = challenge_callback
        
        episode_data = []  # Store (player, state, action, reward_placeholder)
        
        done = False

        while not done:
            current_player = game.current_player
            agent = agent1 if current_player == 0 else agent2
            
            # get current state from current player's perspective
            state = get_state_for_player(game, current_player, p0_hist, p1_hist)
            
            # get valid moves from game and convert to agent action space
            valid_game_moves = game.get_valid_moves(current_player)
            valid_actions = [move_to_action[move] for move in valid_game_moves]
            
            # select action
            action, mode_used = agent.select_action(state, training=True, valid_actions=valid_actions)
            
            # store pre-action state for reward calculation
            game_state_pre = game.get_state()
            pre_coins = game_state_pre['players_coins'][current_player]
            pre_opp_coins = game_state_pre['players_coins'][1 - current_player]
            pre_inf = game_state_pre['influence'][current_player]
            pre_opp_inf = game_state_pre['influence'][1 - current_player]
            
            # execute action in game
            game_move = action_to_move[action]
            update_action_history(p0_hist, p1_hist, current_player, game_move)
            game.play_move(current_player, game_move)
            
            # calculate immediate reward from current player's perspective
            reward = calculate_reward(game, current_player, pre_coins, pre_opp_coins, pre_inf, pre_opp_inf)
            
            # check if game is over
            done = game.is_game_over()
            turn += 1
            if turn >= MAX_TURNS_PER_GAME:
                done = True
            
            # get next state from same player's perspective
            next_state = get_state_for_player(game, current_player, p0_hist, p1_hist)
            
            # store transition
            agent.store_transition(state, action, reward, next_state, done, mode_used)
            
            # train the main networks
            agent.train_step()
            
            # switch turns only after storing the transition
            if not done:
                game.current_player = (game.current_player + 1) % 2

        agent1.train_block()
        agent2.train_block()
        agent1.train_challenge()
        agent2.train_challenge()

        # track wins
        game_state_final = game.get_state()
        if game_state_final['influence'][0] > game_state_final['influence'][1]:
            wins_agent1 += 1
        elif game_state_final['influence'][1] > game_state_final['influence'][0]:
            wins_agent2 += 1

        if ep % 100 == 0:
            win_rate_1 = wins_agent1 / 100
            win_rate_2 = wins_agent2 / 100
            print(f"Round {ep}/{num_episodes} | Win rate p1: {win_rate_1:.2%} | win rate p2: {win_rate_2:.2%}")
            wins_agent1 = 0
            wins_agent2 = 0

    # save the agent that won more games overall
    print(f"Training complete!")
    final_win_rate_1 = wins_agent1 / (wins_agent1 + wins_agent2) if (wins_agent1 + wins_agent2) > 0 else 0.0
    final_win_rate_2 = wins_agent2 / (wins_agent1 + wins_agent2) if (wins_agent1 + wins_agent2) > 0 else 0.0

    if final_win_rate_1 >= final_win_rate_2:
        agent1.save(f"{save_prefix}_best_agent")
    else:
        agent2.save(f"{save_prefix}_best_agent")


if __name__ == "__main__":
    self_play_training(num_episodes=5000, save_prefix="nfsp_coup_demo", pretrain=True)