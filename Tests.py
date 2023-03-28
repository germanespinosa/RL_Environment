import time

from Model import Model
from Agent import Agent, AgentAction
from cellworld import *


class RandomAgent(Agent):
    def get_action(self, observation: dict) -> tuple:
        # print(observation)
        return AgentAction(.05, .15)


w = World.get_from_parameters_names("hexagonal", "canonical", "21_05")
m = Model(w, freq=10)

print (m.is_valid_location(Location(0,1)))

a1 = RandomAgent()
a2 = RandomAgent()

m.add_agent("agent1", a1, Location(0, .5), 0, "r")
m.add_agent("agent2", a2, Location(1, .5), math.pi, "g")
m.run()
print("starting")
while True:
    m.show()
m.stop()
