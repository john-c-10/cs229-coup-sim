import random
import itertools
import asyncio

#cards are 1-5
# 0 is Duke
# 1 is assassin
# 2 is ambassador
# 3 is captain
# 4 is contessa

#valid moves:
# - tax 0
# - assassinate 1
# - exchange 2
# - steal 3
# - income 4
# - foreign aid 5
# - coup 6

#can block:
# - foreign aid 0 (blocked by Duke)
# - assassinate 1 (blocked by Contessa)
# - steal 2 (blocked by Captain or Ambassador)

class CoupGame:
    def __init__(self):
        #coins
        self.players = [2, 2]
        #influence
        self.influence = [2, 2]
        #deck
        self.deck = [i for _ in range(3) for i in range(5)]
        random.shuffle(self.deck)
        #roles
        self.roles = []
        self.new_round()
        #discard pile
        self.discard_pile = [0 for i in range(5)]
        #current turn
        self.current_player = 0
        #callbacks for agent decisions
        self.block_callback = None
        self.challenge_callback = None

    def new_round(self):
        self.roles = []
        for _ in range(2):
            player_roles = [0, 0, 0, 0, 0]  # 5-vector
            for _ in range(2):  # draw 2 cards per player
                card = self.deck.pop(0)
                player_roles[card] += 1
            self.roles.append(player_roles)
        return

    #remove random influence from a player
    def remove_influence(self, player):
        if self.influence[player] <= 0:
            return
        self.influence[player] -= 1
        # find indices where the player has influence (count > 0)
        available_roles = [i for i, count in enumerate(self.roles[player]) if count > 0]
        # randomly pick one of these roles to lose
        removed_role = random.choice(available_roles)
        self.roles[player][removed_role] -= 1
        self.discard_pile[removed_role] += 1
        return

    #check if a move is valid for the current player
    def is_valid_move(self, player, move):
        coins = self.players[player]
        # must coup if >= 10 coins
        if coins >= 10 and move != 6:
            return False
        # can't coup with < 7 coins
        if move == 6 and coins < 7:
            return False
        # can't assassinate with < 3 coins
        if move == 1 and coins < 3:
            return False
        return True

    def play_move(self, player, move):
        # tax 0 (Duke, cannot be blocked, gain 3 coins)
        # assassinate 1 (Assassin, can be blocked by Contessa, costs 3 coins)
        # exchange 2 (Ambassador, cannot be blocked, exchange cards)
        # steal 3 (Captain, can be blocked by Captain/Ambassador, steal 2 coins)
        # income 4 (No role required, cannot be blocked, gain 1 coin)
        # foreign aid 5 (No role required, can be blocked by Duke, gain 2 coins)
        # coup 6 (No role required, cannot be blocked, costs 7 coins, remove influence)
        
        #default to income if the move is invalid
        if not self.is_valid_move(player, move):
            move = 4
        
        opponent = (player + 1) % 2

        # tax, gain 3 coins (claims duke)
        if move == 0:
            lie_result = self.predict_lie(player, 0)  # 0 = Duke
            if lie_result == "lie":
                return False  # player caught lying, lost influence
            elif lie_result == "no lie":
                self.players[player] += 3 # challenge and no lie
                return True
            self.players[player] += 3
            return True
        
        # assassinate, can be blocked by contessa, costs 3 coins
        elif move == 1:
            self.players[player] -= 3
            lie_result = self.predict_lie(player, 1)  # 1 = Assassin
            if lie_result == "lie":
                return False  # player caught lying
            elif lie_result == "no lie":
                self.remove_influence(opponent)
                return True
            
            # check if opponent blocks
            successful_block = self.check_blocks(opponent, 1)
            if not successful_block:
                self.remove_influence(opponent)
            return True

        # exchange, cannot be blocked (claims ambassador)
        elif move == 2:
            lie_result = self.predict_lie(player, 2)  # 2 = Ambassador
            if lie_result == "lie":
                return False
            elif lie_result == "no lie" or not lie_result:
                self.pick_cards(player)
            return True

        # steal, can be blocked by Captain or Ambassador, steal 2 coins
        elif move == 3:
            lie_result = self.predict_lie(player, 3)  # 3 = Captain
            if lie_result == "lie":
                return False
            elif lie_result == "no lie":
                stolen = min(2, self.players[opponent])
                self.players[opponent] -= stolen
                self.players[player] += stolen
                return True

            # Check if opponent blocks
            successful_block = self.check_blocks(opponent, 2)  # 2 = steal block
            if not successful_block:
                stolen = min(2, self.players[opponent])
                self.players[opponent] -= stolen
                self.players[player] += stolen
            return True
        
        # income, gain 1 coin (no claim, cannot be blocked)
        elif move == 4:
            self.players[player] += 1
            return True
        
        # foreign aid, gain 2 coins, can be blocked by Duke
        elif move == 5:
            successful_block = self.check_blocks(opponent, 0)  # 0 = foreign aid block
            if not successful_block:
                self.players[player] += 2
            return True
        
        # coup, costs 7 coins, cannot be blocked
        elif move == 6:
            self.players[player] -= 7
            self.remove_influence(opponent)
            return True
        
        return False

    #ambassador simplified to just pick the top 2 cards off the top of the deck
    def pick_cards(self, player):
        # put the cards back in the deck
        for card, count in enumerate(self.roles[player]):
            for _ in range(count):
                self.deck.append(card)
                self.roles[player][card] -= 1
        # draw the first 2 cards from the deck and assign to player
        new_roles = [0] * 5
        for _ in range(2):
            if self.deck:
                drawn_card = self.deck.pop(0)
                new_roles[drawn_card] += 1
        self.roles[player] = new_roles
        return

    def check_blocks(self, player, block_type):
        #check if the player blocks
        # 0: foreign aid (blocked by Duke)
        # 1: assassinate (blocked by contessa) 
        # 2: steal (blocked by Captain or Ambassador)

        #decide if the player wants to claim a block
        claim = self.select_action_claim(player,block_type)
        if not claim:
            return True
        
        # Determine which card they're claiming to block with
        if block_type == 0:  # foreign aid => Duke
            claimed_card = 0
        elif block_type == 1:  # assassinate => contess
            claimed_card = 4
        elif block_type == 2:  # steal => Captain or Ambassador (choose cptn)
            claimed_card = [3, 2]
        else:
            return True
        
        #check if the claim is challenged
        opponent = (player + 1) % 2
        lie_result = self.predict_lie(player, claimed_card)
        if lie_result == "lie":
            return False
        elif lie_result == "no lie":
            return True
        return True
        # challenge = self.select_action_lie(opponent, claimed_card)
        
        # if challenge:
        #     if self.roles[player][claimed_card] > 0:
        #         self.remove_influence(opponent)
        #         self.deck.append(claimed_card)
        #         self.roles[player][claimed_card] -= 1
        #         random.shuffle(self.deck)
        #         new_card = self.deck.pop(0)
        #         self.roles[player][new_card] += 1
        #         return True
        #     else:
        #         self.remove_influence(player)
        #         return False
        # else:
        #     return True

    def predict_lie(self, player, claimed_card):
        #check if the lie is called
        opponent = (player + 1) % 2
        #opponent decides whether to challenge
        challenge = self.select_action_lie(opponent, claimed_card)
        card_in_hand = None
        if isinstance(claimed_card, list) or isinstance(claimed_card, tuple):
            claimed_card_1 = claimed_card[0]
            claimed_card_2 = claimed_card[1]
            has_card_1 = self.roles[player][claimed_card_1] > 0
            has_card_2 = self.roles[player][claimed_card_2] > 0
            if has_card_1:
                card_in_hand = claimed_card_1
            elif has_card_2:
                card_in_hand = claimed_card_2
        else:
            has_card = self.roles[player][claimed_card] > 0
            if has_card:
                card_in_hand = claimed_card
        if challenge:
            #check if they have the correct card (deal with captain and ambassador so we have to check both)
            if card_in_hand:
                #player has the card, challenger loses influence
                self.remove_influence(opponent)
                #player shows card and draws a new one
                self.deck.append(card_in_hand)
                self.roles[player][card_in_hand] -= 1
                random.shuffle(self.deck)
                new_card = self.deck.pop(0)
                self.roles[player][new_card] += 1
                return "no lie"
            else:
                #player was lying, loses influence
                self.remove_influence(player)
                return "lie"
        else:
            #not challenged
            return False
    
    def select_action_claim(self, player, block_type):
        if self.block_callback is not None:
            return self.block_callback(player, block_type)
        return False
    
    def select_action_lie(self, player, claimed_card):
        if self.challenge_callback is not None:
            return self.challenge_callback(player, claimed_card)
        return False
    
    def get_valid_moves(self, player):
        #return list of valid move indices for the player
        valid_moves = [i for i in range(7)]
        if self.players[player] >= 10:
            valid_moves = [6]
        if self.players[player] < 7:
            valid_moves.remove(6)
        if self.players[player] < 3:
            valid_moves.remove(1)
        return valid_moves
    
    def is_game_over(self):
        #check if the game is over
        return self.influence[0] == 0 or self.influence[1] == 0
    
    def get_winner(self):
        #return the winner (0 or 1) or None if game not over
        if self.influence[0] == 0:
            return 1
        elif self.influence[1] == 0:
            return 0
        return None
    
    def get_state(self):
        #return the current game state as a dictionary
        return {
            'players_coins': self.players.copy(),
            'influence': self.influence.copy(),
            'roles': [r.copy() for r in self.roles],
            'deck_size': len(self.deck),
            'discard_pile': self.discard_pile,
            'current_player': self.current_player
        }

def print_game_state(game):
    #print the current game state
    print("GAME STATE: ")
    card_names = ["Duke", "Assassin", "Ambassador", "Captain", "Contessa"]
    for i in range(2):
        print(f"\nPlayer {i}:")
        print(f"  Coins: {game.players[i]}")
        print(f"  Influence: {game.influence[i]}")
        print(f"  Roles: {[card_names[j] for j, count in enumerate(game.roles[i]) for _ in range(count)]}")
    print(f"Discard pile: {[card_names[c] for c in range(5) for _ in range(game.discard_pile[c])]}")

def select_move(game, player, move):
    #TODO: model to do this
    valid_moves = game.get_valid_moves(player)
    coins = game.players[player]
    
    #must coup if >= 10 coins
    if coins >= 10:
        return 6
    
    # heuristic to coup if we can and opponent has 1 influence
    opponent = (player + 1) % 2
    if 6 in valid_moves and game.influence[opponent] == 1:
        return 6
    
    return random.choice(valid_moves)

def main():
    #run game
    print("Starting Coup Game...")
    game = CoupGame()
    
    move_names = ["Tax", "Assassinate", "Exchange", "Steal", "Income", "Foreign Aid", "Coup"]
    
    print_game_state(game)
    
    turn = 0
    max_turns = 50
    
    while not game.is_game_over() and turn < max_turns:
        current_player = game.current_player
        print(f"\nTurn {turn + 1} - Player {current_player}'s turn")
        
        legal_moves = game.get_valid_moves(current_player)
        move = select_move(game, current_player, legal_moves)
        #select and play a move
        print(f"Player {current_player} plays: {move_names[move]}")
        
        result = game.play_move(current_player, move)
        print(f"Move {'successful' if result else 'failed'}")
        
        print_game_state(game)
        
        #switch turns
        game.current_player = (game.current_player + 1) % 2
        turn += 1
        
        #check for game over
        if game.is_game_over():
            winner = game.get_winner()
            print(f"Player {winner} wins")
            break
    
    if turn >= max_turns:
        print("\nGame ended due to turn limit")

if __name__ == "__main__":
    main()