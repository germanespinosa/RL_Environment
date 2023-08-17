# RL_Environment

## Low Entropy 10%

```{python}
from gym_env import Environment
# e = 1: Entropy 10%
env = Environment(e = 1, freq=100, has_predator=True, max_step=250, env_type = "train", predator_speed = 1.0)
env.reset()

# Random
env.step(env.action_space.sample())

# show render
env.show()

# close
env.close()
```

<img width="600" alt="image" src="https://github.com/germanespinosa/RL_Environment/assets/80494218/41e82bb5-8887-4ade-8996-07bdab04329d">




## High Entorpy 90%

```{python}
from gym_env import Environment
env = Environment(e = 9, freq=100, has_predator=True, max_step=250, env_type = "train")
env.reset()
# Random
env.step(env.action_space.sample())

# show render
env.show()

# close
env.close()
```
![image](https://github.com/germanespinosa/RL_Environment/assets/80494218/4fb5dcbc-e9f1-4aae-bdab-59c115fdba89)
