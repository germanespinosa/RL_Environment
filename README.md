# RL_Environment

## Low Entropy 10%

```{python}
from gym_env import Environment
env = Environment("16_01", freq=100, has_predator=False, max_step=250)
env.reset()
```

<img width="600" alt="image" src="https://github.com/germanespinosa/RL_Environment/assets/80494218/600ca0ca-fdb8-4d55-93f4-104dd06e387b">



## High Entorpy 90%

```{python}
from gym_env import Environment
env = Environment("16_09", freq=100, has_predator=False, max_step=250)
env.reset()
```
<img width="600" alt="image" src="https://github.com/germanespinosa/RL_Environment/assets/80494218/186e9b8c-e4d8-4011-87d9-594046e257f6">
