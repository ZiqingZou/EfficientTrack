from Open_Env import RLEnv
import numpy as np
from PPO_Agent import PPOAgent, ReplayBuffer
from concurrent.futures import ThreadPoolExecutor
import threading
from copy import deepcopy


def merge_buffers(*buffers):
    # Assume all buffers have the same state_dim and action_dim
    state_dim = buffers[0].state_pool.size(1)
    action_dim = buffers[0].action_pool.size(1)
    merged_capacity = sum(buffer.num_push for buffer in buffers)
    merged_buffer = ReplayBuffer(merged_capacity, state_dim, action_dim)

    current_index = 0
    for buffer in buffers:
        end_index = current_index + buffer.num_push

        merged_buffer.state_pool[current_index:end_index] = buffer.state_pool[:buffer.num_push]
        merged_buffer.action_pool[current_index:end_index] = buffer.action_pool[:buffer.num_push]
        merged_buffer.next_state_pool[current_index:end_index] = buffer.next_state_pool[:buffer.num_push]
        merged_buffer.reward_pool[current_index:end_index] = buffer.reward_pool[:buffer.num_push]
        merged_buffer.done_pool[current_index:end_index] = buffer.done_pool[:buffer.num_push]
        merged_buffer.advantage_pool[current_index:end_index] = buffer.advantage_pool[:buffer.num_push]
        merged_buffer.log_prob_old_pool[current_index:end_index] = buffer.log_prob_old_pool[:buffer.num_push]
        merged_buffer.td_target_pool[current_index:end_index] = buffer.td_target_pool[:buffer.num_push]

        merged_buffer.start_index.extend(idx + current_index for idx in buffer.start_index)
        merged_buffer.end_index.extend(idx + current_index for idx in buffer.end_index)

        current_index = end_index

    merged_buffer.num_push = merged_capacity
    return merged_buffer


def run_episodes(agent, env_new, lock, episode_avg_rewards, steps_per_episode, buffer_list):
    buffer = ReplayBuffer(buffer_size=600, state_dim=332, action_dim=4)
    for _ in range(1):
        total_reward = 0
        s = env_new.reset()
        done = False
        step = 0
        while not done:
            a = np.array(agent.train(s))
            s_, r, done, _ = env_new.step(a)
            buffer.push(s, a, s_, r, done, start=bool(step == 0), end=done)
            total_reward += r
            step += 1
            s = s_
        with lock:
            episode_avg_rewards.append(total_reward / step)
            steps_per_episode.append(step)
    with lock:
        buffer_list.append(buffer)


if __name__ == "__main__":
    agent = PPOAgent(state_dim=332, action_dim=4)
    # agent.load(epoch=2000)
    env = RLEnv()
    lock = threading.Lock()

    for i in range(0, 10000):
        episode_avg_rewards = []
        steps_per_episode = []
        buffer_list = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(run_episodes, agent, deepcopy(env),
                                       lock, episode_avg_rewards, steps_per_episode, buffer_list) for _ in range(8)]
            for future in futures:
                future.result()

        total_avg_reward = np.mean(episode_avg_rewards)
        total_step = np.sum(steps_per_episode)
        total_buffer = merge_buffers(*buffer_list)

        agent.writer.add_scalar('Return/avg_return', total_avg_reward, global_step=i)
        print('iter: ', i + 1)
        print('Data number: %d' % total_buffer.num_push)
        print("*************TRAIN**************")
        agent.update(total_buffer)
        print("***********TRAIN OVER***********")

        if (i + 1) % 10 == 0:
            agent.save(i + 1)  # save params after 10*n update

        episode_avg_rewards.clear()
        steps_per_episode.clear()
        buffer_list.clear()
