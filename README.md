# RL_Environment

## Low Entropy 10%

```{python}
from prey_env import gymnasium_Environment_C as Environment
from prey_env import gymnasium_Environment_D as Environment
from prey_env import gym_Environment_D as Environment

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
from prey_env import gymnasium_Environment_C as Environment
from prey_env import gymnasium_Environment_D as Environment
from prey_env import gym_Environment_D as Environment
env = Environment(e = 9, freq=100, has_predator=True, max_step=250, env_type = "train")
env.reset()
# Random
env.step(env.action_space.sample())

# show render
env.show()

# close
env.close()
```
<img width="600" alt="image" src="https://github.com/germanespinosa/RL_Environment/assets/80494218/238fe286-262b-4de7-bb17-187460bedb08">




