from environment import PushEnv


SLEEP = False
env = PushEnv(gui=1, seed=None)

it = 0
while True:
    action = it % 4
    env.step(action, sleep=SLEEP)
    env.reset_objects()
    it += 1
