import os
import sys
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import time
import json

#actual training sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    DEVICE,
    NUM_EPISODES,
    MAX_STEPS_PER_EPISODE,
    WARMUP_STEPS,
    TRAIN_FREQ,
    EVAL_FREQ,
    EVAL_GAMES,
    LOG_FREQ,
    SAVE_FREQ,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_MAIN_ACTIONS,
)
from .agent import DQNAgent, DQNAgentForBaseline
from .env import CoupEnv, SelfPlayEnv, make_env, make_self_play_env
from coup_baseline.baseline import RandomBot, NoLieBot, HeuristicBot, MCTSAgent

class TrainingMetrics:
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
        self.training_losses = []
        self.td_errors = []
        self.action_counts = defaultdict(int)
        
    def add_episode(
        self,
        reward: float,
        length: int,
        won: bool,
        actions: List[int],
    ):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.wins.append(won)
        for action in actions:
            self.action_counts[action] += 1
    
    def add_training_step(self, loss: float, td_error: float):
        self.training_losses.append(loss)
        self.td_errors.append(td_error)
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_wins = self.wins[-window:]
        recent_losses = self.training_losses[-window:] if self.training_losses else [0]
        
        return {
            "mean_reward": np.mean(recent_rewards) if recent_rewards else 0,
            "mean_length": np.mean(recent_lengths) if recent_lengths else 0,
            "win_rate": np.mean(recent_wins) if recent_wins else 0,
            "mean_loss": np.mean(recent_losses) if recent_losses else 0,
        }
    
    def clear(self):
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.wins.clear()
        self.training_losses.clear()
        self.td_errors.clear()
        self.action_counts.clear()

class EvaluationResults:
    
    def __init__(self):
        self.results = defaultdict(lambda: {
            "wins": 0,
            "games": 0,
            "influence_margin": [],
            "coin_margin": [],
            "action_counts": defaultdict(int),
            "challenge_attempts": 0,
            "challenge_successes": 0,
            "block_attempts": 0,
            "block_successes": 0,
        })
    
    def add_game(
        self,
        opponent_name: str,
        won: bool,
        influence_margin: int,
        coin_margin: int,
        actions: List[int],
        challenge_success: Optional[bool] = None,
        block_success: Optional[bool] = None,
    ):
        r = self.results[opponent_name]
        r["games"] += 1
        if won:
            r["wins"] += 1
        r["influence_margin"].append(influence_margin)
        r["coin_margin"].append(coin_margin)
        for action in actions:
            r["action_counts"][action] += 1
        if challenge_success is not None:
            r["challenge_attempts"] += 1
            if challenge_success:
                r["challenge_successes"] += 1
        if block_success is not None:
            r["block_attempts"] += 1
            if block_success:
                r["block_successes"] += 1
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, r in self.results.items():
            games = r["games"]
            if games == 0:
                continue
            summary[name] = {
                "win_rate": r["wins"] / games,
                "games": games,
                "mean_influence_margin": np.mean(r["influence_margin"]),
                "mean_coin_margin": np.mean(r["coin_margin"]),
                "challenge_success_rate": (
                    r["challenge_successes"] / r["challenge_attempts"]
                    if r["challenge_attempts"] > 0 else 0
                ),
                "block_success_rate": (
                    r["block_successes"] / r["block_attempts"]
                    if r["block_attempts"] > 0 else 0
                ),
            }
        return summary
    
    def clear(self):
        self.results.clear()

# self-play ep (only player 0 stores transitions)
def run_self_play_episode(
    agent: DQNAgent,
    env: SelfPlayEnv,
) -> Tuple[float, int, bool, List[int]]:
    obs, legal_mask, current_player = env.reset()
    
    hidden_states = [agent.online_net.init_hidden(1), agent.online_net.init_hidden(1)]
    
    total_reward = 0.0
    actions_taken = []
    step = 0
    done = False
    
    agent.episode_buffer.clear()
    
    while not done and step < MAX_STEPS_PER_EPISODE:
        obs = env.get_observation_for_player(current_player)
        legal_mask = env.get_legal_mask_for_player(current_player)
        
        agent.hidden_state = hidden_states[current_player]
        action, new_hidden = agent.select_action(obs, legal_mask, training=True)
        hidden_states[current_player] = new_hidden
        
        next_obs, reward, done, next_legal_mask, info, next_player = env.step(action)
        
        if current_player == 0:
            agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                legal_mask=legal_mask,
                info=info,
            )
            total_reward += reward
            actions_taken.append(action)
        
        current_player = next_player if not done else current_player
        step += 1
    
    agent.end_episode()
    
    won = env.env.get_winner() == 0
    
    return total_reward, step, won, actions_taken

# eval game vs. baseline opponent
def run_evaluation_game(
    agent: DQNAgent,
    env: CoupEnv,
    opponent,
    agent_player: int = 0,
) -> Tuple[bool, int, int, List[int]]:
    obs, legal_mask = env.reset(agent_player=agent_player)
    
    agent.reset_hidden_state()
    actions_taken = []
    step = 0
    done = False
    
    while not done and step < MAX_STEPS_PER_EPISODE:
        current_player = env.game.current_player
        
        if current_player == agent_player:
            obs = env._get_observation()
            legal_mask = env._get_legal_mask()
            action = agent.get_policy_action(obs, legal_mask)
            actions_taken.append(action)
        else:
            action = opponent.select_action(env.game, current_player)
        
        next_obs, reward, done, next_legal_mask, info = env.step(action)
        step += 1
    
    opponent_player = (agent_player + 1) % 2
    influence_margin = env.game.influence[agent_player] - env.game.influence[opponent_player]
    coin_margin = env.game.players[agent_player] - env.game.players[opponent_player]
    won = env.get_winner() == agent_player
    
    return won, influence_margin, coin_margin, actions_taken

def evaluate_against_baselines(
    agent: DQNAgent,
    num_games: int = EVAL_GAMES,
) -> EvaluationResults:
    results = EvaluationResults()
    
    baselines = [
        ("RandomBot", RandomBot()),
        ("NoLieBot", NoLieBot()),
        ("HeuristicBot", HeuristicBot()),
        ("MCTSAgent", MCTSAgent(num_simulations=20, depth_limit=3)),
    ]
    
    env = make_env()
    
    for name, opponent in baselines:
        games_for_this = num_games if name != "MCTSAgent" else max(10, num_games // 10)
        
        for game_idx in range(games_for_this):
            agent_player = game_idx % 2
            
            won, inf_margin, coin_margin, actions = run_evaluation_game(
                agent, env, opponent, agent_player
            )
            
            results.add_game(
                opponent_name=name,
                won=won,
                influence_margin=inf_margin,
                coin_margin=coin_margin,
                actions=actions,
            )
    
    return results

# main loop: collect eps, train after warmup, log/eval/save
def train(
    num_episodes: int = NUM_EPISODES,
    checkpoint_dir: str = CHECKPOINT_DIR,
    resume_from: Optional[str] = None,
):
    print(f"Training DQN agent for {num_episodes} episodes")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    agent = DQNAgent()
    env = make_self_play_env()
    
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        agent.load(resume_from)
        try:
            start_episode = int(resume_from.split("_")[-1].split(".")[0])
        except:
            pass
    
    metrics = TrainingMetrics()
    
    total_steps = 0
    start_time = time.time()
    
    for episode in range(start_episode, num_episodes):
        reward, length, won, actions = run_self_play_episode(agent, env)
        metrics.add_episode(reward, length, won, actions)
        total_steps += length
        
        if total_steps > WARMUP_STEPS and total_steps % TRAIN_FREQ == 0:
            train_result = agent.train_step()
            if train_result:
                metrics.add_training_step(train_result["loss"], train_result["td_error"])
        
        if (episode + 1) % LOG_FREQ == 0:
            stats = metrics.get_recent_stats()
            elapsed = time.time() - start_time
            eps_per_sec = (episode - start_episode + 1) / elapsed
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"\nWin rate: {stats['win_rate']:.2%}")
            print(f"Mean reward: {stats['mean_reward']:.3f}")
            print(f"Mean length: {stats['mean_length']:.1f}")
            print(f"Mean loss: {stats['mean_loss']:.4f}")
            print(f"Epsilon: {agent.current_epsilon:.3f}")
            print(f"Buffer size: {len(agent.replay_buffer)}")
            print(f"Speed: {eps_per_sec:.1f} eps/sec")
        
        if (episode + 1) % EVAL_FREQ == 0:
            print("\n*** Evaluation ***")
            eval_results = evaluate_against_baselines(agent)
            summary = eval_results.get_summary()
            
            for name, stats in summary.items():
                print(f"\nvs {name}:")
                print(f"\nWin rate: {stats['win_rate']:.2%} ({int(stats['games'])} games)")
                print(f"Influence margin: {stats['mean_influence_margin']:.2f}")
                print(f"Coin margin: {stats['mean_coin_margin']:.2f}")
            print()
            
            eval_path = os.path.join(checkpoint_dir, f"eval_{episode + 1}.json")
            with open(eval_path, "w") as f:
                json.dump(summary, f, indent=2)
        
        if (episode + 1) % SAVE_FREQ == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    final_path = os.path.join(checkpoint_dir, "dqn_final.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")
    
    print("\n****** Final Evaluation ******")
    eval_results = evaluate_against_baselines(agent, num_games=EVAL_GAMES * 2)
    summary = eval_results.get_summary()
    
    for name, stats in summary.items():
        print(f"\nvs {name}:")
        print(f"\nWin rate: {stats['win_rate']:.2%}")
        print(f"Influence margin: {stats['mean_influence_margin']:.2f}")
        print(f"Coin margin: {stats['mean_coin_margin']:.2f}")
        if stats.get("challenge_success_rate", 0) > 0:
            print(f"Challenge success: {stats['challenge_success_rate']:.2%}")
        if stats.get("block_success_rate", 0) > 0:
            print(f"Block success: {stats['block_success_rate']:.2%}")
    
    return agent

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN agent for Coup")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to train")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train(
        num_episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )

if __name__ == "__main__":
    main()