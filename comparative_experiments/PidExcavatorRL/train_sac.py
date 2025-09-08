from Open_Env import RLEnv
import numpy as np

from SAC_Agent import SACAgent


if __name__ == "__main__":
    agent = SACAgent(state_dim=332, action_dim=4)
    env = RLEnv()

    # agent.load(epoch=0)

    for i in range(0, 10000):
        # new episode
        s = env.reset()
        done = False
        avg_total_reward = 0
        step = 0

        while not done:
            a = np.array(agent.train(s))
            s_, r, done, _ = env.step(a)
            agent.replay_buffer.push(s, a, s_, r, done)
            avg_total_reward += r
            step += 1
            s = s_

        avg_total_reward = avg_total_reward / step  # push dates to buffer
        agent.writer.add_scalar('Return/avg_return', avg_total_reward, global_step=i)
        print('iter: ', i + 1)
        print('Total data number: %d' % agent.replay_buffer.num_push)

        print("*************TRAIN**************")
        if i < 100:
            agent.update(i)
        elif i < 500:
            for j in range(10):
                agent.update(i)
        elif i < 1000:  # 3000
            for j in range(20):
                agent.update(i)
        else:
            for j in range(50):
                agent.update(i)
        print("***********TRAIN OVER***********")

        if (i+1) % 10 == 0:
            agent.save(i+1)  # save params after 10*n update
            # env.save(i+1)  # save log data
