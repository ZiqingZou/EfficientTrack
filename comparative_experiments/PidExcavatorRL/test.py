from Open_Env import RLEnv
import numpy as np
type = 'SAC'
if type == 'SAC':
    from SAC_Agent import SACAgent
elif type == 'PPO':
    from PPO_Agent import PPOAgent


def main():
    if type == 'SAC':
        agent = SACAgent(state_dim=332, action_dim=4)
    elif type == 'PPO':
        agent = PPOAgent(state_dim=332, action_dim=4)
    agent.load(epoch=550, name=None)
    env = RLEnv()

    for i in range(1, 9):
        s = env.reset(i, 'test')
        done = False
        while not done:
            a = np.array(agent.test(s))
            s, _, done, _ = env.step(a)

        env.save_test(fp='SAC_Open')


if __name__ == '__main__':
    main()
