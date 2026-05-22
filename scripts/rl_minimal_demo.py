import random


# 最小强化学习 Demo：Q-learning 学会在一维世界里走到终点
# 世界：0 -- 1 -- 2 -- 3 -- 4
# 起点：0
# 终点：4
# 动作：0 表示向左，1 表示向右

NUM_STATES = 5
START_STATE = 0
GOAL_STATE = 4
ACTIONS = [0, 1]

EPISODES = 200
MAX_STEPS = 20
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPSILON = 0.2


def step(state, action):
    """执行一个动作，返回：下一个状态、奖励、是否结束。"""
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(GOAL_STATE, state + 1)

    if next_state == GOAL_STATE:
        reward = 1
        done = True
    else:
        reward = -0.01
        done = False

    return next_state, reward, done


def choose_action(q_table, state):
    """epsilon-greedy：大多数时候选当前最优动作，少数时候随机探索。"""
    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    left_value = q_table[state][0]
    right_value = q_table[state][1]
    return 0 if left_value > right_value else 1


def train():
    """训练 Q 表。"""
    q_table = [[0.0 for _ in ACTIONS] for _ in range(NUM_STATES)]

    for episode in range(EPISODES):
        state = START_STATE
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = choose_action(q_table, state)
            next_state, reward, done = step(state, action)

            old_q = q_table[state][action]
            best_next_q = max(q_table[next_state])

            # Q-learning 核心公式：
            # 新Q值 = 旧Q值 + 学习率 * (当前奖励 + 折扣因子 * 下一状态最大Q值 - 旧Q值)
            q_table[state][action] = old_q + LEARNING_RATE * (reward + DISCOUNT * best_next_q - old_q)

            state = next_state
            total_reward += reward

            if done:
                break

        if (episode + 1) % 50 == 0:
            print(f"第 {episode + 1:3d} 轮训练，累计奖励：{total_reward:.2f}")

    return q_table


def test(q_table):
    """使用训练好的 Q 表走一遍。"""
    state = START_STATE
    path = [state]

    for _ in range(MAX_STEPS):
        action = 0 if q_table[state][0] > q_table[state][1] else 1
        state, _, done = step(state, action)
        path.append(state)

        if done:
            break

    return path


if __name__ == "__main__":
    q_table = train()

    print("\n训练后的 Q 表：")
    for state, values in enumerate(q_table):
        print(f"状态 {state}: 向左={values[0]:.3f}, 向右={values[1]:.3f}")

    path = test(q_table)
    print("\n测试路径：", " -> ".join(map(str, path)))
