import numpy as np
import random
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import CoupGame
from .config import (
    NUM_MAIN_ACTIONS,
    NUM_ACTIONS,
    ACTIONS,
    HISTORY_LENGTH,
    ACTION_ENCODING_DIM,
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

# maps main action to claimed card (None if no claim)
ACTION_TO_CLAIM = {
    "INCOME": None,
    "FOREIGN_AID": None,
    "TAX": 0, # Duke
    "STEAL": 3, # Captain
    "EXCHANGE": 2, # Ambassador
    "ASSASSINATE": 1, # Assassin
    "COUP": None,
}

# maps block action to claimed card
BLOCK_TO_CLAIM = {
    7: 0, # BLOCK_FOREIGN_AID means claiming Duke
    8: 4, # BLOCK_ASSASSINATION means claiming Contessa
    9: 3, # BLOCK_STEAL_CAPTAIN means claiming Captain
    10: 2, # BLOCK_STEAL_AMBASSADOR means claiming Ambassador
}

class CoupEnv:
    
    def __init__(self, opponent_agent=None):
        self.game = None
        self.opponent_agent = opponent_agent
        
        self.agent_player = 0
        
        self.action_history = []
        
        self.turn_count = 0

        self.phase = "main"
        
        self.pending_action = None # main action index (0-6)
        self.pending_action_name = None # action name string
        self.pending_actor = None # who did the main action
        self.pending_target = None # target of the action
        self.pending_claim = None # card claimed by main action
        self.pending_block = None # block action index (7-10) if blocked
        self.pending_block_claim = None # card claimed by blocker
        
        self.current_info_bundle = None
        
    def reset(self, agent_player: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        self.game = CoupGame()
        self.agent_player = agent_player
        self.action_history = []
        self.turn_count = 0
        self.phase = "main"
        self._clear_pending()
        
        obs = self._get_observation()
        legal_mask = self._get_legal_mask()
        
        return obs, legal_mask
    
    def _clear_pending(self):
        self.pending_action = None
        self.pending_action_name = None
        self.pending_actor = None
        self.pending_target = None
        self.pending_claim = None
        self.pending_block = None
        self.pending_block_claim = None
        self.current_info_bundle = None
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict[str, Any]]:
        acting_player = self.game.current_player
        step_inf_before = self.game.influence.copy()
        step_coins_before = self.game.players.copy()
        
        # create or reuse info bundle (tracks turn-level info)
        if self.current_info_bundle is None:
            self.current_info_bundle = InfoBundle(
                influence_before=step_inf_before,
                coins_before=step_coins_before,
                actor=acting_player,
            )
        
        info_bundle = self.current_info_bundle
        
        if self.phase == "main":
            _, done, info_bundle = self._process_main_action(action, info_bundle)
        elif self.phase == "challenge_action":
            _, done, info_bundle = self._process_challenge_decision(action, info_bundle)
        elif self.phase == "block":
            _, done, info_bundle = self._process_block_decision(action, info_bundle)
        elif self.phase == "challenge_block":
            _, done, info_bundle = self._process_block_challenge_decision(action, info_bundle)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
        
        # capture state after this step
        step_inf_after = self.game.influence.copy()
        step_coins_after = self.game.players.copy()
        
        info_bundle.influence_after = step_inf_after
        info_bundle.coins_after = step_coins_after
        info_bundle.done = done
        
        winner = None
        if done:
            winner = self.game.get_winner()
            info_bundle.winner = winner
        
        reward = self._compute_reward(
            acting_player=acting_player,
            done=done,
            winner=winner,
        )
        
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
    
    def _compute_reward(
        self,
        acting_player: int,
        done: bool,
        winner: Optional[int],
    ) -> float:
        if done:
            if winner == acting_player:
                return 1.0
            elif winner is not None:
                return -1.0
            else:
                # draw
                return 0.0
        
        # all intermediate steps get zero reward
        return 0.0
    
    # handles main action, may transition to challenge/block phases
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
        info_bundle.actor = current_player
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
        
        # store pending action info
        self.pending_action = action
        self.pending_action_name = action_name
        self.pending_actor = current_player
        self.pending_target = opponent
        self.pending_claim = ACTION_TO_CLAIM.get(action_name)
        
        reward = 0.0
        done = False
        
        # actions that execute immediately (no challenge/block possible)
        if action_name == "INCOME":
            self.game.players[current_player] += 1
            info_bundle.result = "succeeded"
            self._advance_turn()
            done = self.game.is_game_over()
            return reward, done, info_bundle
            
        if action_name == "COUP":
            self.game.players[current_player] -= 7
            self.game.remove_influence(opponent)
            info_bundle.result = "succeeded"
            self._advance_turn()
            done = self.game.is_game_over()
            return reward, done, info_bundle
        
        # foreign aid which can be blocked but not challenged
        if action_name == "FOREIGN_AID":
            if self.game.influence[opponent] > 0:
                self.phase = "block"
                self.game.current_player = opponent  # opponent decides
            else:
                # opponent eliminated, action succeeds
                self.game.players[current_player] += 2
                info_bundle.result = "succeeded"
                self._advance_turn()
            done = self.game.is_game_over()
            return reward, done, info_bundle
        
        # tax, steal, exchange, assassinate which can be challenged
        if action_name == "ASSASSINATE":
            self.game.players[current_player] -= 3  # pay cost upfront
        
        if self.game.influence[opponent] > 0:
            self.phase = "challenge_action"
            self.game.current_player = opponent  # opponent decides
        else:
            # opponent eliminated, execute action
            self._execute_pending_action(info_bundle)
            self._advance_turn()
        
        done = self.game.is_game_over()
        return reward, done, info_bundle
    
    def _process_challenge_decision(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        reward = 0.0
        done = False
        
        challenger = self.game.current_player
        actor = self.pending_actor
        claimed_card = self.pending_claim
        
        # action 12 = challenge, action 13 = accept
        is_challenge = (action == 12)
        
        self._add_to_history(action, challenger, actor)
        
        if is_challenge:
            has_card = self.game.roles[actor][claimed_card] > 0
            
            if has_card:
                # challenge failed so challenger loses influence
                self.game.remove_influence(challenger)
                new_card = self._reveal_and_replace_card(actor, claimed_card)
                
                info_bundle.challenge_info = {
                    "challenger": challenger,
                    "claim_challenged": claimed_card,
                    "challenge_success": False,
                    "who_lost_influence": challenger,
                    "revealed_card": claimed_card,
                }
                
                # check if game over from losing influence
                if self.game.is_game_over():
                    info_bundle.result = "succeeded"
                    self._advance_turn()
                    return reward, True, info_bundle
                
                # action can proceed socheck if blockable
                if self._is_action_blockable() and self.game.influence[challenger] > 0:
                    self.phase = "block"
                    self.game.current_player = self.pending_target  # target decides
                else:
                    # not blockable or target eliminated so carry out action
                    self._execute_pending_action(info_bundle)
                    self._advance_turn()
            else:
                # challenge succeeded so actor was lying
                self.game.remove_influence(actor)
                info_bundle.result = "challenged"
                info_bundle.challenge_info = {
                    "challenger": challenger,
                    "claim_challenged": claimed_card,
                    "challenge_success": True,
                    "who_lost_influence": actor,
                }
                self._advance_turn()
        else:
            # accepted (no challenge) so check if blockable
            if self._is_action_blockable() and self.game.influence[self.pending_target] > 0:
                self.phase = "block"
                self.game.current_player = self.pending_target  # target decides
            else:
                # not blockable so carry out action
                self._execute_pending_action(info_bundle)
                self._advance_turn()
        
        done = self.game.is_game_over()
        return reward, done, info_bundle
    
    def _process_block_decision(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        reward = 0.0
        done = False
        
        blocker = self.game.current_player
        actor = self.pending_actor
        
        # action 11 = DECLINE_BLOCK, 7-10 = block variants
        is_block = (action >= 7 and action <= 10)
        
        self._add_to_history(action, blocker, actor)
        
        if is_block:
            self.pending_block = action
            self.pending_block_claim = BLOCK_TO_CLAIM.get(action)
            
            info_bundle.block_info = {
                "blocker": blocker,
                "block_action": action,
                "block_claim": self.pending_block_claim,
            }
            
            # actor can challenge the block
            if self.game.influence[actor] > 0:
                self.phase = "challenge_block"
                self.game.current_player = actor  # actor decides
            else:
                # actor eliminated which meansblock succeeds automatically
                info_bundle.result = "blocked"
                info_bundle.block_info["block_success"] = True
                self._advance_turn()
        else:
            # declined to block which means execute action
            self._execute_pending_action(info_bundle)
            self._advance_turn()
        
        done = self.game.is_game_over()
        return reward, done, info_bundle
    
    def _process_block_challenge_decision(
        self,
        action: int,
        info_bundle: InfoBundle,
    ) -> Tuple[float, bool, InfoBundle]:
        reward = 0.0
        done = False
        
        challenger = self.game.current_player  # original actor
        blocker = self.pending_target
        block_claim = self.pending_block_claim
        
        # action 12 = challenge, action 13 = accept
        is_challenge = (action == 12)
        
        self._add_to_history(action, challenger, blocker)
        
        if is_challenge:
            has_card = self.game.roles[blocker][block_claim] > 0
            
            if has_card:
                # challenge failed which means challenger (actor) loses influence
                self.game.remove_influence(challenger)
                new_card = self._reveal_and_replace_card(blocker, block_claim)
                
                info_bundle.block_info["block_success"] = True
                info_bundle.block_info["block_challenge_outcome"] = "blocker_wins"
                info_bundle.result = "blocked"
                info_bundle.challenge_info = {
                    "challenger": challenger,
                    "claim_challenged": block_claim,
                    "challenge_success": False,
                    "who_lost_influence": challenger,
                    "revealed_card": block_claim,
                }
            else:
                # challenge succeeded which means blocker was lying
                self.game.remove_influence(blocker)
                
                info_bundle.block_info["block_success"] = False
                info_bundle.block_info["block_challenge_outcome"] = "challenger_wins"
                info_bundle.challenge_info = {
                    "challenger": challenger,
                    "claim_challenged": block_claim,
                    "challenge_success": True,
                    "who_lost_influence": blocker,
                }
                
                # block failed which means execute original action (if target still alive)
                if self.game.influence[self.pending_target] > 0 or self.pending_action_name not in ["STEAL", "ASSASSINATE"]:
                    self._execute_pending_action(info_bundle)
                else:
                    info_bundle.result = "succeeded"
            
            self._advance_turn()
        else:
            # accepted the block (no challenge)
            info_bundle.result = "blocked"
            info_bundle.block_info["block_success"] = True
            self._advance_turn()
        
        done = self.game.is_game_over()
        return reward, done, info_bundle
    
    def _is_action_blockable(self) -> bool:
        return self.pending_action_name in ["FOREIGN_AID", "STEAL", "ASSASSINATE"]
    
    def _execute_pending_action(self, info_bundle: InfoBundle):
        actor = self.pending_actor
        target = self.pending_target
        action_name = self.pending_action_name
        
        if action_name == "FOREIGN_AID":
            self.game.players[actor] += 2
            info_bundle.result = "succeeded"
            
        elif action_name == "TAX":
            self.game.players[actor] += 3
            info_bundle.result = "succeeded"
            
        elif action_name == "STEAL":
            if self.game.influence[target] > 0:
                steal_amt = min(2, self.game.players[target])
                self.game.players[actor] += steal_amt
                self.game.players[target] -= steal_amt
                info_bundle.steal_transfer_amount = steal_amt
            info_bundle.result = "succeeded"
            
        elif action_name == "EXCHANGE":
            self.game.pick_cards(actor)
            info_bundle.result = "succeeded"
            
        elif action_name == "ASSASSINATE":
            # cost already paid in main action
            if self.game.influence[target] > 0:
                self.game.remove_influence(target)
            info_bundle.result = "succeeded"
    
    # honest defender shows card, shuffles back, draws new
    def _reveal_and_replace_card(self, player: int, card: int) -> int:
        self.game.roles[player][card] -= 1
        
        self.game.deck.append(card)
        random.shuffle(self.game.deck)
        
        new_card = self.game.deck.pop(0)
        self.game.roles[player][new_card] += 1
        
        return new_card
    
    def _advance_turn(self):
        # find next alive player after pending_actor
        next_player = (self.pending_actor + 1) % 2
        if self.game.influence[next_player] <= 0:
            next_player = self.pending_actor
        
        self.game.current_player = next_player
        self.turn_count += 1
        self.phase = "main"
        self._clear_pending()
    
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
            history_vec[base_idx + NUM_ACTIONS] = float(entry["actor"])  # 0 or 1
            history_vec[base_idx + NUM_ACTIONS + 1] = float(entry["target"]) if entry["target"] is not None else 0.5
        
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
        info["phase"] = self.env.phase
        
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
