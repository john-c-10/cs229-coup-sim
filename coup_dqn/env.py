import numpy as np
import random
import copy
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import CoupGame
from .config import (
    NUM_CARD_TYPES,
    NUM_MAIN_ACTIONS,
    NUM_BLOCK_ACTIONS,
    NUM_CHALLENGE_ACTIONS,
    NUM_ACTIONS,
    ALL_ACTIONS,
    ACTIONS,
    BLOCK_ACTIONS,
    CHALLENGE_ACTIONS,
    HISTORY_LENGTH,
    ACTION_ENCODING_DIM,
    OBS_DIM,
    MAX_STEPS_PER_EPISODE,
    USE_TIME_PENALTY,
    TIME_PENALTY,
    STALL_THRESHOLD,
)

@dataclass
class ActionResult:
    success: bool = True
    blocked: bool = False
    challenged: bool = False
    challenge_succeeded: bool = False
    blocker: Optional[int] = None
    block_type: Optional[str] = None
    challenger: Optional[int] = None
    who_lost_influence: Optional[int] = None
    revealed_card: Optional[int] = None
    steal_amount: int = 0

@dataclass
class InfoBundle:
    done: bool = False
    winner: Optional[int] = None
    
    influence_before: List[int] = field(default_factory=lambda: [0, 0])
    influence_after: List[int] = field(default_factory=lambda: [0, 0])
    
    coins_before: List[int] = field(default_factory=lambda: [0, 0])
    coins_after: List[int] = field(default_factory=lambda: [0, 0])
    
    action_type: str = ""
    actor: int = 0
    target: Optional[int] = None
    result: str = "succeeded" # "succeeded", "blocked", "challenged"
    
    block_info: Dict[str, Any] = field(default_factory=dict)
    challenge_info: Dict[str, Any] = field(default_factory=dict)
    
    steal_transfer_amount: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "done": self.done,
            "winner": self.winner,
            "influence_before": self.influence_before.copy(),
            "influence_after": self.influence_after.copy(),
            "coins_before": self.coins_before.copy(),
            "coins_after": self.coins_after.copy(),
            "action_type": self.action_type,
            "actor": self.actor,
            "target": self.target,
            "result": self.result,
            "block_info": self.block_info.copy(),
            "challenge_info": self.challenge_info.copy(),
            "steal_transfer_amount": self.steal_transfer_amount,
        }

class CoupEnv:
    
    def __init__(self, opponent_agent=None):
        self.game = None
        self.opponent_agent = opponent_agent
        
        self.agent_player = 0
        
        self.action_history = []
        
        self.turn_count = 0

        # multi-phase to be made
        self.phase = "main"
        
        self.pending_action = None
        self.pending_claim = None
        self.pending_block = None
        
    def reset(self, agent_player: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        self.game = CoupGame()
        self.agent_player = agent_player
        self.action_history = []
        self.turn_count = 0
        self.phase = "main"
        self.pending_action = None
        self.pending_claim = None
        self.pending_block = None
        
        obs = self._get_observation()
        legal_mask = self._get_legal_mask()
        
        return obs, legal_mask
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict[str, Any]]:
        info_bundle = InfoBundle(
            influence_before=self.game.influence.copy(),
            coins_before=self.game.players.copy(),
            actor=self.game.current_player,
        )
        
        current_player = self.game.current_player
        opponent = (current_player + 1) % 2
        
        if self.phase == "main":
            reward, done, info_bundle = self._process_main_action(action, info_bundle)
        elif self.phase == "challenge_action":
            reward, done, info_bundle = self._process_challenge_decision(action, info_bundle)
        elif self.phase == "block":
            reward, done, info_bundle = self._process_block_decision(action, info_bundle)
        elif self.phase == "challenge_block":
            reward, done, info_bundle = self._process_block_challenge_decision(action, info_bundle)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
        
        info_bundle.influence_after = self.game.influence.copy()
        info_bundle.coins_after = self.game.players.copy()
        info_bundle.done = done
        
        if done:
            info_bundle.winner = self.game.get_winner()
        
        if USE_TIME_PENALTY and self.turn_count > STALL_THRESHOLD:
            reward += TIME_PENALTY
        
        next_obs = self._get_observation()
        legal_mask = self._get_legal_mask()
        
        return next_obs, reward, done, legal_mask, info_bundle.to_dict()
    
    def _is_action_valid(self, player: int, action_name: str) -> bool:
        coins = self.game.players[player]
        
        if coins >= 10:
            return action_name == "COUP"
        
        if action_name == "COUP" and coins < 7:
            return False
        
        if action_name == "ASSASSINATE" and coins < 3:
            return False
        
        return True
    
    # handles all 7 main actions with challenge/block
    def _process_main_action(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        current_player = self.game.current_player
        opponent = (current_player + 1) % 2
        coins = self.game.players[current_player]
        
        action = min(action, NUM_MAIN_ACTIONS - 1)
        action_name = ACTIONS[action]
        
        info_bundle.action_type = action_name
        info_bundle.target = opponent if action_name in ["STEAL", "ASSASSINATE", "COUP"] else None
        
        if not self._is_action_valid(current_player, action_name):
            # if coins >= 10, player MUST coup
            if coins >= 10:
                action = 6
                action_name = "COUP"
                info_bundle.action_type = action_name
                info_bundle.target = opponent
            else:
                action = 0
                action_name = "INCOME"
                info_bundle.action_type = action_name
                info_bundle.target = None
        
        self._add_to_history(action, current_player, opponent)
        
        reward = 0.0
        done = False
        
        if action_name == "INCOME":
            self.game.players[current_player] += 1
            info_bundle.result = "succeeded"
            self._advance_turn()
            
        elif action_name == "FOREIGN_AID":
            blocked = self._check_opponent_block(opponent, block_type=0)
            if blocked:
                info_bundle.result = "blocked"
                info_bundle.block_info = {"blocker": opponent, "block_type": "Duke", "block_success": True}
            else:
                self.game.players[current_player] += 2
                info_bundle.result = "succeeded"
            self._advance_turn()
            
        elif action_name == "TAX":
            challenged = self._check_opponent_challenge(opponent, claimed_card=0)
            if challenged:
                has_card = self.game.roles[current_player][0] > 0
                if has_card:
                    self.game.remove_influence(opponent)
                    new_card = self._reveal_and_replace_card(current_player, 0)
                    self.game.players[current_player] += 3
                    info_bundle.result = "succeeded"
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "claim_challenged": "Duke",
                        "challenge_success": False,
                        "who_lost_influence": opponent,
                        "revealed_card": 0,
                    }
                else:
                    self.game.remove_influence(current_player)
                    info_bundle.result = "challenged"
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "claim_challenged": "Duke",
                        "challenge_success": True,
                        "who_lost_influence": current_player,
                    }
            else:
                self.game.players[current_player] += 3
                info_bundle.result = "succeeded"
            self._advance_turn()
            
        elif action_name == "STEAL":
            challenged = self._check_opponent_challenge(opponent, claimed_card=3)
            if challenged:
                has_card = self.game.roles[current_player][3] > 0
                if has_card:
                    self.game.remove_influence(opponent)
                    new_card = self._reveal_and_replace_card(current_player, 3)
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "challenge_success": False,
                        "who_lost_influence": opponent,
                        "revealed_card": 3,
                    }
                    blocked = self._check_opponent_block(opponent, block_type=2)
                    if blocked:
                        info_bundle.result = "blocked"
                        info_bundle.block_info = {"blocker": opponent, "block_success": True}
                    else:
                        steal_amt = min(2, self.game.players[opponent])
                        self.game.players[current_player] += steal_amt
                        self.game.players[opponent] -= steal_amt
                        info_bundle.steal_transfer_amount = steal_amt
                        info_bundle.result = "succeeded"
                else:
                    self.game.remove_influence(current_player)
                    info_bundle.result = "challenged"
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "challenge_success": True,
                        "who_lost_influence": current_player,
                    }
            else:
                blocked = self._check_opponent_block(opponent, block_type=2)
                if blocked:
                    info_bundle.result = "blocked"
                    info_bundle.block_info = {"blocker": opponent, "block_success": True}
                else:
                    steal_amt = min(2, self.game.players[opponent])
                    self.game.players[current_player] += steal_amt
                    self.game.players[opponent] -= steal_amt
                    info_bundle.steal_transfer_amount = steal_amt
                    info_bundle.result = "succeeded"
            self._advance_turn()
            
        elif action_name == "EXCHANGE":
            challenged = self._check_opponent_challenge(opponent, claimed_card=2)
            if challenged:
                has_card = self.game.roles[current_player][2] > 0
                if has_card:
                    self.game.remove_influence(opponent)
                    new_card = self._reveal_and_replace_card(current_player, 2)
                    self.game.pick_cards(current_player)
                    info_bundle.result = "succeeded"
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "challenge_success": False,
                        "who_lost_influence": opponent,
                        "revealed_card": 2,
                    }
                else:
                    self.game.remove_influence(current_player)
                    info_bundle.result = "challenged"
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "challenge_success": True,
                        "who_lost_influence": current_player,
                    }
            else:
                self.game.pick_cards(current_player)
                info_bundle.result = "succeeded"
            self._advance_turn()
            
        elif action_name == "ASSASSINATE":
            self.game.players[current_player] -= 3
            challenged = self._check_opponent_challenge(opponent, claimed_card=1)
            if challenged:
                has_card = self.game.roles[current_player][1] > 0
                if has_card:
                    self.game.remove_influence(opponent)
                    new_card = self._reveal_and_replace_card(current_player, 1)
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "challenge_success": False,
                        "who_lost_influence": opponent,
                        "revealed_card": 1,
                    }
                    blocked = self._check_opponent_block(opponent, block_type=1)
                    if blocked and self.game.influence[opponent] > 0:
                        info_bundle.result = "blocked"
                        info_bundle.block_info = {"blocker": opponent, "block_success": True}
                    else:
                        if self.game.influence[opponent] > 0:
                            self.game.remove_influence(opponent)
                        info_bundle.result = "succeeded"
                else:
                    self.game.remove_influence(current_player)
                    info_bundle.result = "challenged"
                    info_bundle.challenge_info = {
                        "challenger": opponent,
                        "challenge_success": True,
                        "who_lost_influence": current_player,
                    }
            else:
                blocked = self._check_opponent_block(opponent, block_type=1)
                if blocked:
                    info_bundle.result = "blocked"
                    info_bundle.block_info = {"blocker": opponent, "block_success": True}
                else:
                    self.game.remove_influence(opponent)
                    info_bundle.result = "succeeded"
            self._advance_turn()
            
        elif action_name == "COUP":
            self.game.players[current_player] -= 7
            self.game.remove_influence(opponent)
            info_bundle.result = "succeeded"
            self._advance_turn()
        
        done = self.game.is_game_over()
        
        return reward, done, info_bundle
    
    # honest defender shows card, shuffles back, draws new
    def _reveal_and_replace_card(self, player: int, card: int) -> int:
        self.game.roles[player][card] -= 1
        
        self.game.deck.append(card)
        random.shuffle(self.game.deck)
        
        new_card = self.game.deck.pop(0)
        self.game.roles[player][new_card] += 1
        
        return new_card
    
    def _check_opponent_challenge(self, opponent: int, claimed_card: int) -> bool:
        if self.opponent_agent is not None:
            return self.opponent_agent.decide_challenge(self.game, opponent, claimed_card)
        return random.random() < 0.2
    
    def _check_opponent_block(self, opponent: int, block_type: int) -> bool:
        if self.game.influence[opponent] <= 0:
            return False
        
        if self.opponent_agent is not None:
            return self.opponent_agent.decide_block(self.game, opponent, block_type)
        return random.random() < 0.3
    
    def _advance_turn(self):
        self.game.current_player = (self.game.current_player + 1) % 2
        self.turn_count += 1
        self.phase = "main"
    
    def _add_to_history(self, action: int, actor: int, target: Optional[int]):
        entry = {
            "action": action,
            "actor": actor,
            "target": target,
        }
        self.action_history.append(entry)
        if len(self.action_history) > HISTORY_LENGTH:
            self.action_history = self.action_history[-HISTORY_LENGTH:]
    
    def _get_observation(self) -> np.ndarray:
        current = self.game.current_player
        opponent = (current + 1) % 2
        
        obs = []
        
        obs.append(self.game.players[current] / 12.0)
        obs.append(self.game.players[opponent] / 12.0)
        obs.append(self.game.influence[current] / 2.0)
        obs.append(self.game.influence[opponent] / 2.0)
        
        for count in self.game.discard_pile:
            obs.append(count / 3.0)
        
        for count in self.game.roles[current]:
            obs.append(count / 2.0)
        
        history_vec = np.zeros(HISTORY_LENGTH * ACTION_ENCODING_DIM)
        for i, entry in enumerate(self.action_history[-HISTORY_LENGTH:]):
            base_idx = i * ACTION_ENCODING_DIM
            if entry["action"] < NUM_ACTIONS:
                history_vec[base_idx + entry["action"]] = 1.0
            history_vec[base_idx + NUM_ACTIONS] = entry["actor"]
            history_vec[base_idx + NUM_ACTIONS + 1] = entry["target"] if entry["target"] is not None else -1
        
        obs.extend(history_vec.tolist())
        
        obs.append(min(self.turn_count / MAX_STEPS_PER_EPISODE, 1.0))
        
        phase_map = {"main": 0, "challenge_action": 1, "block": 2, "challenge_block": 3}
        obs.append(phase_map.get(self.phase, 0) / 3.0)
        
        return np.array(obs, dtype=np.float32)
    
    # computes valid actions based on coins and phase
    def _get_legal_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        current = self.game.current_player
        coins = self.game.players[current]
        
        if self.phase == "main":
            if coins >= 10:
                mask[6] = True # coup
            else:
                mask[0] = True # income
                mask[1] = True # foreign aid
                mask[2] = True # tax
                mask[3] = True # steal
                mask[4] = True # exchange
                
                if coins >= 3:
                    mask[5] = True # assassinate
                
                if coins >= 7:
                    mask[6] = True # coup
                    
        elif self.phase == "challenge_action" or self.phase == "challenge_block":
            mask[12] = True # challenge
            mask[13] = True # accept
            
        elif self.phase == "block":
            mask[11] = True # decline block
            
            if self.pending_action == 1: # foreign aid
                mask[7] = True # block foreign aid
            elif self.pending_action == 5: # assassinate
                mask[8] = True # block assassination
            elif self.pending_action == 3: # steal
                mask[9] = True # block steal captain
                mask[10] = True # block steal ambassador
        
        if not mask.any():
            if coins >= 10:
                mask[6] = True
            else:
                mask[0] = True
        
        return mask
    
    def _process_challenge_decision(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        # placeholder for multi-phase right now
        reward = 0.0
        done = self.game.is_game_over()
        self._advance_turn()
        return reward, done, info_bundle
    
    def _process_block_decision(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        # placeholder for multi-phase right now
        reward = 0.0
        done = self.game.is_game_over()
        self._advance_turn()
        return reward, done, info_bundle
    
    def _process_block_challenge_decision(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        # placeholder for multi-phase right now
        reward = 0.0
        done = self.game.is_game_over()
        self._advance_turn()
        return reward, done, info_bundle
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "game_state": self.game.get_state(),
            "agent_player": self.agent_player,
            "turn_count": self.turn_count,
            "phase": self.phase,
            "action_history": self.action_history.copy(),
        }
    
    def is_game_over(self) -> bool:
        return self.game.is_game_over() if self.game else True
    
    def get_winner(self) -> Optional[int]:
        return self.game.get_winner() if self.game else None


class SelfPlayEnv:
    
    def __init__(self):
        self.env = CoupEnv()
        self.agent_perspectives = [None, None]
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, int]:
        obs, legal_mask = self.env.reset(agent_player=0)
        return obs, legal_mask, self.env.game.current_player
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict[str, Any], int]:
        current_player = self.env.game.current_player
        
        next_obs, reward, done, legal_mask, info = self.env.step(action)
        
        info["acting_player"] = current_player
        
        next_player = self.env.game.current_player if not done else -1
        
        return next_obs, reward, done, legal_mask, info, next_player
    
    def get_observation_for_player(self, player: int) -> np.ndarray:
        original = self.env.game.current_player
        self.env.game.current_player = player
        obs = self.env._get_observation()
        self.env.game.current_player = original
        return obs
    
    def get_legal_mask_for_player(self, player: int) -> np.ndarray:
        original = self.env.game.current_player
        self.env.game.current_player = player
        mask = self.env._get_legal_mask()
        self.env.game.current_player = original
        return mask

def make_env(opponent_agent=None) -> CoupEnv:
    return CoupEnv(opponent_agent=opponent_agent)

def make_self_play_env() -> SelfPlayEnv:
    return SelfPlayEnv()
