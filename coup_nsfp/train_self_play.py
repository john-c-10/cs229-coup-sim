from agent import NFSPAgent
from env_simple_coup import SimpleCoup1v1Env


#actually training

def self_play_training(num_episodes: int = 10_000, save_prefix: str = "nfsp_coup"):
    env = SimpleCoup1v1Env()
    state_dim = env._get_state_for_player(1).shape[0]

    agent1 = NFSPAgent(state_dim)
    agent2 = NFSPAgent(state_dim)

    wins_agent1 = 0
    wins_agent2 = 0

    for ep in range(1, num_episodes + 1):
        state = env.reset() #reset game to starting positions
        done = False

        while not done: #done defines when the game is over and you don't need to consider future states
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2

            action, mode_used = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done, mode_used)

            #only happens when you have big enough buffer
            agent.train_step()

            state = next_state

        # crude win tracking by influence count
        if env.p1_influence > env.p2_influence:
            wins_agent1 += 1
        elif env.p2_influence > env.p1_influence:
            wins_agent2 += 1

        if ep % 100 == 0:
            print(f"Episode {ep}/{num_episodes} | Wins A1: {wins_agent1} | Wins A2: {wins_agent2}")

    # Save agent1
    agent1.save(save_prefix)


if __name__ == "__main__":
    # Adjust num_episodes as needed
    self_play_training(num_episodes=2000, save_prefix="nfsp_coup_demo")
