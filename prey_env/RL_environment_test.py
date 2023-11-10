from gymnasium_env_bins import Environment

if __name__ == "__main__":
    Env = Environment()
    observations, infos = Env.reset()
    for i in range(100000):
        Env.step(Env.action_space.sample())
        Env.show()
        if i%10==0:
            Env.reset()