# test MCTS baseline against random, no-lie, and heuristic bots

import types
import random
from coup_baseline.baseline import MCTSAgent, RandomBot, NoLieBot, HeuristicBot
from collections import defaultdict
from game import CoupGame

MOVE_NAMES = ["Tax", "Assassinate", "Exchange", "Steal", "Income", "Foreign Aid", "Coup"]

# simulator that runs games with agent hooks and tracks metrics for evaluation
class CardTrackingSimulator:
    def __init__(self, seed = None):
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
        self.challenge_tracker = {'made': [0, 0], 'successful': [0, 0]}
    
    # agent decision methods for game instance to override default behavior
    def _attach_agent_hooks_with_tracking(self, game, agent0, agent1):
        challenge_tracker = self.challenge_tracker
        
        def select_action_claim_wrapper(player, block_type):
            if player == 0:
                return agent0.decide_block(game, player, block_type)
            else:
                return agent1.decide_block(game, player, block_type)
        
        def select_action_lie_wrapper(player, claimed_card):
            if player == 0:
                return agent0.decide_challenge(game, player, claimed_card)
            else:
                return agent1.decide_challenge(game, player, claimed_card)
        
        original_predict_lie = game.predict_lie
        
        def predict_lie_wrapper(self_instance, player, claimed_card):
            opponent = (player + 1) % 2
            challenge = select_action_lie_wrapper(opponent, claimed_card)
            
            if challenge:
                challenge_tracker['made'][opponent] += 1
                card_in_hand = None
                if isinstance(claimed_card, list) or isinstance(claimed_card, tuple):
                    claimed_card_1 = claimed_card[0]
                    claimed_card_2 = claimed_card[1]
                    has_card_1 = self_instance.roles[player][claimed_card_1] > 0
                    has_card_2 = self_instance.roles[player][claimed_card_2] > 0
                    if has_card_1:
                        card_in_hand = claimed_card_1
                    elif has_card_2:
                        card_in_hand = claimed_card_2
                else:
                    has_card = self_instance.roles[player][claimed_card] > 0
                    if has_card:
                        card_in_hand = claimed_card
                
                if card_in_hand:
                    self_instance.remove_influence(opponent)
                    self_instance.deck.append(card_in_hand)
                    self_instance.roles[player][card_in_hand] -= 1
                    import random
                    random.shuffle(self_instance.deck)
                    new_card = self_instance.deck.pop(0)
                    self_instance.roles[player][new_card] += 1
                    return "no lie"
                else:
                    challenge_tracker['successful'][opponent] += 1
                    self_instance.remove_influence(player)
                    return "lie"
            else:
                return False
        
        game.select_action_claim = types.MethodType(
            lambda self, player, block_type: select_action_claim_wrapper(player, block_type),
            game
        )
        game.select_action_lie = types.MethodType(
            lambda self, player, claimed_card: select_action_lie_wrapper(player, claimed_card),
            game
        )
        game.predict_lie = types.MethodType(predict_lie_wrapper, game)
    
    def play_game_with_tracking(self, agent0, agent1, max_turns = 100, track_predictions = False):
        game = CoupGame()
        self.challenge_tracker = {'made': [0, 0], 'successful': [0, 0]}
        self._attach_agent_hooks_with_tracking(game, agent0, agent1)
        
        initial_influence = [game.influence[0], game.influence[1]]
        initial_coins = [game.players[0], game.players[1]]
        
        action_counts = [defaultdict(int), defaultdict(int)]
        cards_removed = [0, 0]
        prediction_accuracy = [0.0, 0.0]
        prediction_count = [0, 0]
        
        turn = 0
        
        while not game.is_game_over() and turn < max_turns:
            current_player = game.current_player
            opponent = (current_player + 1) % 2
            
            if track_predictions:
                agent = agent0 if current_player == 0 else agent1
                if hasattr(agent, 'predict_opponent_hand'):
                    predicted_hand = agent.predict_opponent_hand(game, opponent)
                    actual_hand = game.roles[opponent].copy()
                    
                    # calculate l1 distance accuracy (normalized) for hand prediction
                    l1_distance = sum(abs(predicted_hand[i] - actual_hand[i]) for i in range(5))
                    max_distance = sum(actual_hand) * 2
                    if max_distance > 0:
                        accuracy = 1.0 - (l1_distance / max_distance)
                        prediction_accuracy[current_player] += accuracy
                        prediction_count[current_player] += 1
            
            action = agent0.select_action(game, current_player) if current_player == 0 else agent1.select_action(game, current_player)
            action_counts[current_player][action] += 1
            
            before_influence = [game.influence[0], game.influence[1]]
            result = game.play_move(current_player, action)
            after_influence = [game.influence[0], game.influence[1]]
            
            opponent_idx = (current_player + 1) % 2
            cards_lost = before_influence[opponent_idx] - after_influence[opponent_idx]
            if cards_lost > 0:
                cards_removed[current_player] += cards_lost
            
            game.current_player = (game.current_player + 1) % 2
            turn += 1
            
            if game.is_game_over():
                winner = game.get_winner()
                final_coins = [game.players[0], game.players[1]]
                
                per_game_accuracy = [
                    prediction_accuracy[0] / prediction_count[0] if prediction_count[0] > 0 else 0.0,
                    prediction_accuracy[1] / prediction_count[1] if prediction_count[1] > 0 else 0.0
                ]
                
                return {
                    'winner': winner,
                    'turns': turn,
                    'timeout': False,
                    'action_counts': [dict(action_counts[0]), dict(action_counts[1])],
                    'challenges_made': self.challenge_tracker['made'].copy(),
                    'challenges_successful': self.challenge_tracker['successful'].copy(),
                    'cards_removed': cards_removed.copy(),
                    'final_coins': final_coins.copy(),
                    'initial_coins': initial_coins.copy(),
                    'prediction_accuracy': per_game_accuracy,
                    'prediction_turns': prediction_count.copy()
                }
        
        final_coins = [game.players[0], game.players[1]]
        winner = None
        if game.influence[0] > game.influence[1]:
            winner = 0
        elif game.influence[1] > game.influence[0]:
            winner = 1
        elif game.players[0] > game.players[1]:
            winner = 0
        elif game.players[1] > game.players[0]:
            winner = 1
        
        per_game_accuracy = [
            prediction_accuracy[0] / prediction_count[0] if prediction_count[0] > 0 else 0.0,
            prediction_accuracy[1] / prediction_count[1] if prediction_count[1] > 0 else 0.0
        ]
        
        return {
            'winner': winner,
            'turns': turn,
            'timeout': True,
            'action_counts': [dict(action_counts[0]), dict(action_counts[1])],
            'challenges_made': self.challenge_tracker['made'].copy(),
            'challenges_successful': self.challenge_tracker['successful'].copy(),
            'cards_removed': cards_removed.copy(),
            'final_coins': final_coins.copy(),
            'initial_coins': initial_coins.copy(),
            'prediction_accuracy': per_game_accuracy,
            'prediction_turns': prediction_count.copy()
        }
    
    def run_tournament_with_tracking(self, agent0, agent1, num_games = 100, max_turns = 100, track_predictions = False):
        agent0_wins = 0
        agent1_wins = 0
        draws = 0
        total_turns = 0
        total_rewards = [0.0, 0.0]
        total_cards_removed = [0, 0]
        
        action_frequencies = [defaultdict(int), defaultdict(int)]
        total_challenges = [0, 0]
        successful_challenges = [0, 0]
        total_prediction_accuracy = [0.0, 0.0]
        prediction_game_counts = [0, 0]
        
        print()
        for i in range(num_games):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{num_games} games finished")
            result = self.play_game_with_tracking(agent0, agent1, max_turns = max_turns, track_predictions = track_predictions)
            
            if result['winner'] == 0:
                agent0_wins += 1
            elif result['winner'] == 1:
                agent1_wins += 1
            else:
                draws += 1
            
            total_turns += result['turns']
            
            for player in [0, 1]:
                total_rewards[player] += result['final_coins'][player]
                total_cards_removed[player] += result['cards_removed'][player]
                
                for action, count in result['action_counts'][player].items():
                    action_frequencies[player][action] += count
                
                total_challenges[player] += result['challenges_made'][player]
                successful_challenges[player] += result['challenges_successful'][player]
                
                if track_predictions and 'prediction_accuracy' in result and 'prediction_turns' in result:
                    if result['prediction_turns'][player] > 0:
                        total_prediction_accuracy[player] += result['prediction_accuracy'][player]
                        prediction_game_counts[player] += 1
        
        avg_turns = total_turns / num_games if num_games > 0 else 0
        mean_rewards = [total_rewards[0] / num_games, total_rewards[1] / num_games]
        avg_cards_removed = [total_cards_removed[0] / num_games, total_cards_removed[1] / num_games]
        
        action_freq_normalized = [
            {action: count / num_games for action, count in action_frequencies[0].items()},
            {action: count / num_games for action, count in action_frequencies[1].items()}
        ]
        
        # exploitability measures successful challenge rate against predictable opponents
        exploitability = [
            successful_challenges[0] / total_challenges[0] if total_challenges[0] > 0 else 0.0,
            successful_challenges[1] / total_challenges[1] if total_challenges[1] > 0 else 0.0
        ]
        
        avg_prediction_accuracy = [
            total_prediction_accuracy[0] / prediction_game_counts[0] if prediction_game_counts[0] > 0 else 0.0,
            total_prediction_accuracy[1] / prediction_game_counts[1] if prediction_game_counts[1] > 0 else 0.0
        ] if track_predictions else None
        
        return {
            'agent0_wins': agent0_wins,
            'agent1_wins': agent1_wins,
            'draws': draws,
            'num_games': num_games,
            'avg_turns': avg_turns,
            'mean_rewards': mean_rewards,
            'action_frequencies': action_freq_normalized,
            'exploitability': exploitability,
            'avg_cards_removed': avg_cards_removed,
            'challenges_made': total_challenges,
            'challenges_successful': successful_challenges,
            'prediction_accuracy': avg_prediction_accuracy,
            'win_rate_agent0': agent0_wins / num_games if num_games > 0 else 0.0
        }

def test_against_random_bot():
    print("Test 1: MCTS Baseline vs Random Bot")
    
    seed = 25
    mcts = MCTSAgent(num_simulations = 50, depth_limit = 5, seed = seed)
    random_bot = RandomBot(seed = seed + 1000)
    
    simulator = CardTrackingSimulator(seed = seed)
    results = simulator.run_tournament_with_tracking(mcts, random_bot, num_games = 10000, track_predictions = True)
    
    print(f"\nResults after {results['num_games']} games:")
    print(f"\nMCTS Wins: {results['agent0_wins']}")
    print(f"Random Bot Wins: {results['agent1_wins']}")
    print(f"Win Rate: {results['win_rate_agent0']:.2%}")
    print(f"Mean Reward (MCTS): {results['mean_rewards'][0]:.2f} coins")
    print(f"Mean Reward (Random): {results['mean_rewards'][1]:.2f} coins")
    print(f"Average Game Duration: {results['avg_turns']:.1f} turns")
    if results['prediction_accuracy']:
        print(f"Opponent Hand Prediction Accuracy (MCTS): {results['prediction_accuracy'][0]:.2%}")
    
    print(f"\nAction Frequencies (MCTS):")
    print()
    for action, freq in sorted(results['action_frequencies'][0].items()):
        print(f"{MOVE_NAMES[action]}: {freq:.2f} per game")
    print()

def test_against_no_lie_bot():
    print("Test 2: MCTS Baseline vs No-Lie Bot")
    
    seed = 25
    mcts = MCTSAgent(num_simulations = 50, depth_limit = 5, seed = seed)
    no_lie = NoLieBot(seed = seed + 2000)
    
    simulator = CardTrackingSimulator(seed = seed)
    results = simulator.run_tournament_with_tracking(mcts, no_lie, num_games = 10000, track_predictions = True)
    
    print(f"\nResults after {results['num_games']} games:")
    print(f"\nMCTS Wins: {results['agent0_wins']}")
    print(f"No-Lie Bot Wins: {results['agent1_wins']}")
    print(f"Win Rate: {results['win_rate_agent0']:.2%}")
    print(f"Exploitability (MCTS): {results['exploitability'][0]:.2%} successful challenges")
    print(f"Challenges Made (MCTS): {results['challenges_made'][0] / results['num_games']:.2f} per game")
    if results['prediction_accuracy']:
        print(f"Opponent Hand Prediction Accuracy (MCTS): {results['prediction_accuracy'][0]:.2%}")
    
    print(f"\nAction Frequencies (MCTS):")
    print()
    for action, freq in sorted(results['action_frequencies'][0].items()):
        print(f"{MOVE_NAMES[action]}: {freq:.2f} per game")
    
    print(f"\nAction Frequencies (No-Lie Bot):")
    print()
    for action, freq in sorted(results['action_frequencies'][1].items()):
        print(f"{MOVE_NAMES[action]}: {freq:.2f} per game")
    print()

def test_against_heuristic_bot():
    print("Test 3: MCTS Baseline vs Heuristic Bot")
    
    seed = 25
    mcts = MCTSAgent(num_simulations = 50, depth_limit = 5, seed = seed)
    heuristic = HeuristicBot(seed = seed + 3000)
    
    simulator = CardTrackingSimulator(seed = seed)
    results = simulator.run_tournament_with_tracking(mcts, heuristic, num_games = 10000, track_predictions = True)
    
    print(f"\nResults after {results['num_games']} games:")
    print(f"\nMCTS Wins: {results['agent0_wins']}")
    print(f"Heuristic Bot Wins: {results['agent1_wins']}")
    print(f"Win Rate: {results['win_rate_agent0']:.2%}")
    print(f"Mean Reward (MCTS): {results['mean_rewards'][0]:.2f} coins")
    print(f"Mean Reward (Heuristic): {results['mean_rewards'][1]:.2f} coins")
    print(f"Average Game Duration: {results['avg_turns']:.1f} turns")
    if results['prediction_accuracy']:
        print(f"Opponent Hand Prediction Accuracy (MCTS): {results['prediction_accuracy'][0]:.2%}")

    print(f"\nAction Frequencies (MCTS):")
    print()
    for action, freq in sorted(results['action_frequencies'][0].items()):
        print(f"{MOVE_NAMES[action]}: {freq:.2f} per game")
    
    print(f"\nAction Frequencies (Heuristic Bot):")
    print()
    for action, freq in sorted(results['action_frequencies'][1].items()):
        print(f"{MOVE_NAMES[action]}: {freq:.2f} per game")
    print()

def main():
    test_against_random_bot()
    test_against_no_lie_bot()
    test_against_heuristic_bot()

if __name__ == "__main__":
    main()