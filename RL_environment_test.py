import time
from Environment import Environment
from cellworld import *

#creates the environment
e = Environment("21_05", freq=100, has_predator=True)

for i in range(2):
    #runs the environment
    e.run()

    #loops until the predator captures the prey or the prey reaches the goal
    while not e.complete:
        #starts a timer
        t = Timer()
        #reads an observation from the environment.
        observation = e.get_observation()
        #sets an action for the prey
        e.set_action(speed=.2, turning=.2)
        #speed float in habitat lenghts per second.
        #turning float in radians per second
        e.show()
        #computes the remaining time for 1/10 of a second to make the action interval consistent.
        remaining_wait = .1 - t.to_seconds()
        if remaining_wait > 0:
            time.sleep(remaining_wait)

        print(observation)
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

