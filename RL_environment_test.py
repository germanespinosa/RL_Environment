import time
from Environment import Environment
from Agent import Agent, AgentAction
from cellworld import *

call_counter=0

class PreyAgent(Agent):
    def get_action(self, observation: dict) -> AgentAction:
        global call_counter
        call_counter+=1
        if call_counter % 100 == 0:
            print (call_counter)
        #print(observation)
        return AgentAction(.2, .2)

#creates the environment

prey_agent=PreyAgent()
e = Environment("21_05", freq=100, has_predator=True, real_time=False, prey_agent=prey_agent)

for i in range(2):
    #runs the environment
    e.run()

    #loops until the predator captures the prey or the prey reaches the goal
    while not e.complete:
        #starts a timer
        t = Timer()
        #reads an observation from the environment.
        #sets an action for the prey
        #speed float in habitat lenghts per second.
        #turning float in radians per second
        e.show()
        #computes the remaining time for 1/10 of a second to make the action interval consistent.
        remaining_wait = .1 - t.to_seconds()
        if remaining_wait > 0:
            time.sleep(remaining_wait)

        # observation format: Tuple
        # [prey location, prey theta, goal location, predator location, predator theta, captured, goal_reached]
        # prey location: Type Location
        # prey theta: Type float in radians
        # goal location: Type Location
        # predator location: Type Location (None when predator is not visible)
        # predator theta: Type float in radians (None when predator is not visible)
        # captured: Type boolean : Prey has been captured by the predator. environment.complete becomes true.
        # goal_reached : Type boolean : Prey reached the goal location. environment.complete becomes true.

    #stops the environment
    e.stop()

