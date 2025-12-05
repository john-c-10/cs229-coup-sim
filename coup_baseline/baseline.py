# implementation of MCTS, random, no-lie, and heuristic bots

import random
import copy
from typing import List, Optional
from game import CoupGame


class Agent:
    # base interface for Coup game agents
    def select_action(self, game: CoupGame, player: int) -> int:
        # choose the main action to play (0-6)
        raise NotImplementedError
    
    def decide_block(self, game: CoupGame, player: int, block_type: int) -> bool:
        # decide whether to claim a block
        raise NotImplementedError
    
    def decide_challenge(self, game: CoupGame, player: int, claimed_card: int) -> bool:
        # decide whether to challenge a claim
        raise NotImplementedError


class HeuristicBot(Agent):
    # simple agent using rules for opponent modeling

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
    
    def select_action(self, game: CoupGame, player: int) -> int:
        valid_moves = game.get_valid_moves(player)
        coins = game.players[player]
        opponent = (player + 1) % 2
        opponent_coins = game.players[opponent]
        opponent_influence = game.influence[opponent]
        
        # must coup if >= 10 coins
        if coins >= 10:
            return 6
        
        # coup if we can and opponent has 1 influence (game completing move)
        if 6 in valid_moves and opponent_influence == 1:
            return 6
        
        # coup if we can and opponent has high coins (prevent them from couping us)
        if 6 in valid_moves and opponent_coins >= 7:
            return 6
        
        # assassinate if affordable and opponent has 1 influence
        if 5 in valid_moves and opponent_influence == 1:
            return 5
        
        # steal if opponent has coins
        if 3 in valid_moves and opponent_coins >= 2:
            return 3
        
        # tax (Duke) if we have low coins
        if 2 in valid_moves and coins < 5:
            return 2
        
        # foreign aid if we can't tax
        if 1 in valid_moves and coins < 5:
            return 1
        
        # income as fallback
        return 0
    
    def decide_block(self, game: CoupGame, player: int, block_type: int) -> bool:
        opponent = (player + 1) % 2
        
        # block foreign aid with 30% chance if opponent has >3 coins (could have/not have Duke)
        if block_type == 0:
            # heuristic: block if opponent has many coins
            return self.rng.random() < 0.3 if game.players[opponent] > 3 else False
        
        # block assassinate with 40% chance (could have/not have Contessa)
        if block_type == 1:
            return self.rng.random() < 0.4
        
        # block steal with 35% chance (could have/not have Captain/Ambassador)
        if block_type == 2:
            return self.rng.random() < 0.35
        
        return False
    
    def decide_challenge(self, game: CoupGame, player: int, claimed_card: int) -> bool:
        opponent = (player + 1) % 2
        
        # potentially challenge if opponent has many coins
        base_prob = 0.15
        if game.players[opponent] > 5:
            base_prob = 0.25
        
        return self.rng.random() < base_prob

class RandomBot(Agent):
    # agent that selects actions randomly

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
    
    def select_action(self, game: CoupGame, player: int) -> int:
        valid_moves = game.get_valid_moves(player)
        return self.rng.choice(valid_moves)
    
    def decide_block(self, game: CoupGame, player: int, block_type: int) -> bool:
        return self.rng.random() < 0.5
    
    def decide_challenge(self, game: CoupGame, player: int, claimed_card: int) -> bool:
        return self.rng.random() < 0.5


class NoLieBot(Agent):
    # agent that never bluffs and only claims cards it actually has

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
    
    def select_action(self, game: CoupGame, player: int) -> int:
        valid_moves = game.get_valid_moves(player)
        coins = game.players[player]
        opponent = (player + 1) % 2
        
        # must coup if >= 10 coins
        if coins >= 10:
            return 6
        
        # coup if we can and opponent has 1 influence (game completing move)
        if 6 in valid_moves and game.influence[opponent] == 1:
            return 6
        
        # only claim cards we actually have
        # assassinate if we have assassin and opponent has 1 influence
        if 5 in valid_moves and self._has_card(game, player, 1) and game.influence[opponent] == 1:
            return 5
        
        # steal if we have captain and opponent has coins
        if 3 in valid_moves and self._has_card(game, player, 3) and game.players[opponent] >= 2:
            return 3
        
        # tax if we have duke
        if 2 in valid_moves and self._has_card(game, player, 0):
            return 2
        
        # exchange if we have ambassador
        if 4 in valid_moves and self._has_card(game, player, 2):
            return 4
        
        # foreign aid if we can't claim any cards
        if 1 in valid_moves:
            return 1
        
        # income as fallback move
        return 0
    
    def _has_card(self, game: CoupGame, player: int, card: int) -> bool:
        return game.roles[player][card] > 0
    
    def decide_block(self, game: CoupGame, player: int, block_type: int) -> bool:
        # only block if we actually have the required card
        if block_type == 0:  # foreign aid (Duke)
            return self._has_card(game, player, 0)
        elif block_type == 1:  # assassinate (Contessa)
            return self._has_card(game, player, 4)
        elif block_type == 2:  # steal (Captain or Ambassador)
            return self._has_card(game, player, 3) or self._has_card(game, player, 2)
        return False
    
    def decide_challenge(self, game: CoupGame, player: int, claimed_card: int) -> bool:
        # never challenge (this truthful bot assumes others are truthful)
        return False


class MCTSAgent(Agent):
    # rough MCTS with shallow depth and cheap rollouts with heuristic bots

    def __init__(
        self,
        num_simulations: int = 50,
        depth_limit: int = 5,
        exploration_constant: float = 1.4,
        seed: Optional[int] = None
    ):
        self.num_simulations = num_simulations
        self.depth_limit = depth_limit
        self.exploration_constant = exploration_constant
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
        
        # use heuristic agent for opponent modeling and rollouts
        self.opponent_model = HeuristicBot(seed = self.rng.randint(0, 2 ** 31))
        self.rollout_policy = HeuristicBot(seed = self.rng.randint(0, 2 ** 31))
    
    def select_action(self, game: CoupGame, player: int) -> int:
        # select action via MCTS forward search
        
        valid_moves = game.get_valid_moves(player)
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        action_scores = []
        original_rng_state = random.getstate()
        
        for action in valid_moves:
            scores = []
            
            for _ in range(self.num_simulations):
                game_copy = copy.deepcopy(game)
                
                # attach agent hooks to cloned game
                self._attach_agent_hooks(game_copy, player)
                
                try:
                    result = game_copy.play_move(player, action)
                    if not result:
                        # action failed (caught lying), score is bad
                        scores.append(-10.0)
                        continue
                    
                    game_copy.current_player = (game_copy.current_player + 1) % 2
                    
                    score = self._rollout(game_copy, player, depth = 1)
                    scores.append(score)
                except Exception as e:
                    # if simulation fails, use fallback score
                    scores.append(-5.0)
            
            avg_score = sum(scores) / len(scores) if scores else -10.0
            action_scores.append((action, avg_score))
        
        # get back original RNG state after all simulations complete
        random.setstate(original_rng_state)
        
        if action_scores:
            best_action = max(action_scores, key = lambda x: x[1])[0]
            return best_action
        
        # fallback to heuristic
        heuristic = HeuristicBot(seed = self.rng.randint(0, 2 ** 31))
        return heuristic.select_action(game, player)
    
    def _rollout(self, game: CoupGame, root_player: int, depth: int) -> float:
        # run a Monte Carlo rollout from current state (root_player is original player we're optimizing for)

        if game.is_game_over():
            winner = game.get_winner()
            if winner == root_player:
                return 100.0 # win
            else:
                return -100.0 # loss
        
        if depth >= self.depth_limit:
            return self._evaluate_state(game, root_player)
        
        current_player = game.current_player
        
        if current_player == root_player:
            action = self.rollout_policy.select_action(game, current_player)
        else:
            action = self.opponent_model.select_action(game, current_player)
        
        result = game.play_move(current_player, action)
        
        game.current_player = (game.current_player + 1) % 2
        
        return self._rollout(game, root_player, depth + 1)
    
    def _evaluate_state(self, game: CoupGame, player: int) -> float:
        # evaluate a non-game-ending state using heuristics

        opponent = (player + 1) % 2
        
        influence_diff = game.influence[player] - game.influence[opponent]
        
        # coin advantage (scaled down)
        coin_diff = game.players[player] - game.players[opponent]
        
        value = influence_diff * 20.0 + coin_diff * 0.5
        
        # small bonus for having more influence
        if game.influence[player] > game.influence[opponent]:
            value += 5.0
        
        return value
    
    def _attach_agent_hooks(self, game: CoupGame, our_player: int):
        # attach agent decision methods to game instance
        
        def select_action_claim_wrapper(game_instance, player, block_type):
            if player == our_player:
                return self.rollout_policy.decide_block(game_instance, player, block_type)
            else:
                return self.opponent_model.decide_block(game_instance, player, block_type)
        
        def select_action_lie_wrapper(game_instance, player, claimed_card):
            if player == our_player:
                return self.rollout_policy.decide_challenge(game_instance, player, claimed_card)
            else:
                return self.opponent_model.decide_challenge(game_instance, player, claimed_card)
        
        import types
        game.select_action_claim = types.MethodType(
            lambda self, player, block_type: select_action_claim_wrapper(self, player, block_type),
            game
        )
        game.select_action_lie = types.MethodType(
            lambda self, player, claimed_card: select_action_lie_wrapper(self, player, claimed_card),
            game
        )
    
    def decide_block(self, game: CoupGame, player: int, block_type: int) -> bool:
        return self.rollout_policy.decide_block(game, player, block_type)
    
    def decide_challenge(self, game: CoupGame, player: int, claimed_card: int) -> bool:
        return self.rollout_policy.decide_challenge(game, player, claimed_card)
    
    def predict_opponent_hand(self, game: CoupGame, opponent: int) -> List[int]:
        # stopgap: later plug in RL-belief model instead of just card counting
        
        our_player = (opponent + 1) % 2
        
        total_cards_per_type = 3
        deck_size = 15
        
        # track known cards (our hand and discarded pile of cards to estimate what remains in deck)
        our_cards = game.roles[our_player][:]
        discarded = game.discard_pile[:]
        
        our_total = sum(our_cards)
        discarded_total = sum(discarded)
        remaining_pool = deck_size - our_total - discarded_total
        
        if remaining_pool <= 0:
            return [0, 0, 0, 0, 0]
        
        opponent_influence = game.influence[opponent]
        if opponent_influence <= 0:
            return [0, 0, 0, 0, 0]
        
        # calculate probability for each card type based on cards left in deck
        predicted_hand = [0] * 5
        for card_type in range(5):
            total_of_type = total_cards_per_type
            our_count = our_cards[card_type]
            discarded_count = discarded[card_type]
            remaining_of_type = total_of_type - our_count - discarded_count
            remaining_of_type = max(0, remaining_of_type)
            
            if remaining_of_type > 0 and remaining_pool > 0:
                prob_per_card = remaining_of_type / remaining_pool
                predicted_hand[card_type] = prob_per_card * opponent_influence
        
        # normalize predictions to sum up to opponent's actual influence count
        total_predicted = sum(predicted_hand)
        if total_predicted > 0:
            scale_factor = opponent_influence / total_predicted
            predicted_hand = [p * scale_factor for p in predicted_hand]
        
        return predicted_hand