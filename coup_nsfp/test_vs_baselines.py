from agent import NFSPAgent
from game import CoupGame
import numpy as np
from config import ACTIONS, NUM_CARD_TYPES, MAX_TURNS_PER_GAME
from train_self_play import get_state_for_player
from baselineTraining import RandomBot, NoLieBot, HeuristicBot, MCTSAgent


def test_nfsp_vs_baseline(
    nfsp_model_path,
    baseline_agent,
    baseline_name,
    num_games=500,
    nfsp_plays_first=True,
    verbose=False
):
    dummy_game = CoupGame()
    p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    state_dim = get_state_for_player(dummy_game, 0, p0_hist, p1_hist).shape[0]
    
    nfsp_agent = NFSPAgent(state_dim)
    nfsp_agent.load(nfsp_model_path)
    
    action_to_move = {
        0: 4, 1: 5, 2: 0, 3: 3, 4: 2, 5: 1, 6: 6,
    }
    
    # Inverse mapping: game move -> agent action
    move_to_action = {v: k for k, v in action_to_move.items()}
    
    nfsp_player = 0 if nfsp_plays_first else 1
    baseline_player = 1 - nfsp_player
    
    wins_nfsp = 0
    wins_baseline = 0
    total_turns = []
    nfsp_action_counts = {i: 0 for i in range(7)}
    
    print(f"Testing: NFSP vs {baseline_name} ({num_games} games)")
    print(f"NFSP is Player {nfsp_player}, {baseline_name} is Player {baseline_player}")
    
    for game_num in range(1, num_games + 1):
        game = CoupGame()
        p0_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        p1_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        turn = 0
        
        def block_callback(player, block_type):
            if player == nfsp_player:
                current_state = get_state_for_player(game, player, p0_hist, p1_hist)
                block_action, _ = nfsp_agent.select_block_action(current_state, block_type, training=False)
                return block_action == 1
            else:
                return baseline_agent.decide_block(game, player, block_type)
        
        def challenge_callback(player, claimed_card):
            if player == nfsp_player:
                current_state = get_state_for_player(game, player, p0_hist, p1_hist)
                if isinstance(claimed_card, (list, tuple)):
                    card_val = claimed_card[0]
                else:
                    card_val = claimed_card
                challenge_action, _ = nfsp_agent.select_challenge_action(current_state, card_val, training=False)
                return challenge_action == 1
            else:
                return baseline_agent.decide_challenge(game, player, claimed_card)
        
        game.block_callback = block_callback
        game.challenge_callback = challenge_callback
        
        state = get_state_for_player(game, game.current_player, p0_hist, p1_hist)
        done = False
        
        if verbose:
            print(f"Game {game_num}")
        
        while not done and turn < MAX_TURNS_PER_GAME:
            current_player = game.current_player
            
            if current_player == nfsp_player:
                # Get valid moves from game and convert to agent action space
                valid_game_moves = game.get_valid_moves(current_player)
                valid_actions = [move_to_action[move] for move in valid_game_moves]
                
                action, _ = nfsp_agent.select_action(state, training=False, valid_actions=valid_actions)
                nfsp_action_counts[action] += 1
                game_move = action_to_move[action]
            else:
                action = baseline_agent.select_action(game, current_player)
                game_move = action
            
            if verbose:
                move_name = ACTIONS[action] if current_player == nfsp_player else ["Tax", "Assassinate", "Exchange", "Steal", "Income", "Foreign Aid", "Coup"][game_move]
                print(f"Turn {turn + 1}: Player {current_player} plays {move_name}")
            
            game.play_move(current_player, game_move)
            
            done = game.is_game_over()
            game.current_player = (game.current_player + 1) % 2
            turn += 1
            
            state = get_state_for_player(game, game.current_player, p0_hist, p1_hist)
        
        game_state_final = game.get_state()
        if game_state_final['influence'][nfsp_player] > game_state_final['influence'][baseline_player]:
            wins_nfsp += 1
            winner = nfsp_player
        elif game_state_final['influence'][baseline_player] > game_state_final['influence'][nfsp_player]:
            wins_baseline += 1
            winner = baseline_player
        else:
            winner = None
        
        total_turns.append(turn)
        
        if verbose:
            print(f"Winner: Player {winner if winner is not None else 'Draw'} after {turn} turns")
        
        if game_num % 20 == 0 and not verbose:
            print(f"Progress: {game_num}/{num_games} games...")
    
    print(f"\nResults vs {baseline_name}")
    print(f"NFSP wins:       {wins_nfsp:4d} ({wins_nfsp/num_games*100:5.1f}%)")
    print(f"{baseline_name} wins: {wins_baseline:4d} ({wins_baseline/num_games*100:5.1f}%)")
    draws = num_games - wins_nfsp - wins_baseline
    if draws > 0:
        print(f"Draws:           {draws:4d} ({draws/num_games*100:5.1f}%)")
    print(f"Avg game length: {np.mean(total_turns):.1f} turns")
    print(f"\nAction Distribution:")
    total_actions = sum(nfsp_action_counts.values())
    if total_actions > 0:
        for action_idx, count in sorted(nfsp_action_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_actions * 100
            print(f"  {ACTIONS[action_idx]:12s}: {count:4d} ({pct:5.1f}%)")
    
    return {
        'wins_nfsp': wins_nfsp,
        'wins_baseline': wins_baseline,
        'win_rate': wins_nfsp / num_games if num_games > 0 else 0,
        'avg_turns': np.mean(total_turns),
    }


def run_full_evaluation(nfsp_model_path, num_games_per_matchup=100):
    print(f"Full evaluation: {nfsp_model_path}")
    print(f"Testing {num_games_per_matchup} games per matchup")
    
    baselines = [
        (RandomBot(), "RandomBot"),
        (NoLieBot(), "NoLieBot"),
        (HeuristicBot(), "HeuristicBot"),
        (MCTSAgent(num_simulations=30, depth_limit=4), "MCTSAgent(30sim)"),
    ]
    
    results = {}
    
    print("NFSP as Player 0 (goes first)")
    
    for baseline_agent, baseline_name in baselines:
        result = test_nfsp_vs_baseline(
            nfsp_model_path,
            baseline_agent,
            baseline_name,
            num_games=num_games_per_matchup,
            nfsp_plays_first=True,
            verbose=False
        )
        results[f"{baseline_name}_first"] = result
    
    print(f"{'Matchup':<25} {'Position':<10} {'Win Rate':>10} {'Avg Turns':>12}")
    
    for baseline_agent, baseline_name in baselines:
        result_first = results[f"{baseline_name}_first"]
        print(f"{baseline_name:<25} {'First':<10} {result_first['win_rate']*100:>9.1f}% {result_first['avg_turns']:>12.1f}")
    
    all_wins = sum(r['wins_nfsp'] for r in results.values())
    all_losses = sum(r['wins_baseline'] for r in results.values())
    total_games = all_wins + all_losses
    
    print(f"{'Overall':<25} {'First':<10} {all_wins/total_games*100:>9.1f}%")
    print(f"\nTotal games:  {total_games}")
    print(f"Total wins:   {all_wins}")
    print(f"Total losses: {all_losses}")
    
    
    return results


if __name__ == "__main__":
    model_path = "nfsp_coup_demo_best_agent"
    run_full_evaluation(nfsp_model_path=model_path, num_games_per_matchup=500)