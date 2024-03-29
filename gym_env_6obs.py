from Predator import Predator
from Agent import *
from Model import Model
from myPaths import *
from cellworld import *
import random
from gymnasium import Env
from gymnasium import spaces
import numpy as np



class Environment(Env):
    def __init__(self, world_name: str, freq: int = 100, has_predator: bool = False, real_time: bool = False, prey_agent:Agent=None, max_step = 200):
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        self.model = Model(pworld=self.world, freq=freq, real_time=real_time)
        self.goal_location = Location(1, .5)
        self.start_location = Location(0, .5)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.has_predator = has_predator
        self.max_step = max_step
        self.current_step = 0
        self.episode_reward_history = []
        self.current_episode_reward = 0
        self.goal_threshold = self.world.implementation.cell_transformation.size / 2
        self.capture_threshold = self.world.implementation.cell_transformation.size
        self.model.add_agent("prey", prey_agent, Location(0,0), 0, "b", pauto_update=prey_agent is not None)
        self.goal_area = self.model.display.circle(location=self.goal_location,
                                                   color="g",
                                                   alpha=.5,
                                                   radius=self.goal_threshold)
        self.predator = None
        if has_predator:
            paths_builder = Paths_builder.get_from_name("hexagonal", world_name)
            self.predator = Predator(self.world,
                                     ppath_builder=paths_builder,
                                     pvisibility=self.model.visibility,
                                     pP_value=2,
                                     pI_value=0,
                                     pD_value=0,
                                     pmax_speed=.15,
                                     pmax_turning_speed=math.pi)

            self.spawn_locations = Location_list()
            for c in self.world.cells.free_cells():
                if not self.model.visibility.is_visible(self.start_location, c.location):
                    self.spawn_locations.append(c.location)
            self.model.add_agent("predator", self.predator, Location(0, 0), 0, "r")
            self.predator_destination = self.model.display.circle(location=Location(0,0), color="b", alpha=.5, radius=0.01)
            self.predator_destination_cell = self.model.display.circle(location=Location(0,0), color="g", alpha=.5, radius=0.02)
            self.predator_capture_area = self.model.display.circle(location=Location(0,0), color="r", alpha=.5, radius=self.capture_threshold)

    def is_goal_reached(self, prey_location: Location):
        return prey_location.dist(self.goal_location) <= self.goal_threshold

    def is_prey_captured(self, prey_location: Location, predator_location: Location):
        return prey_location.dist(predator_location) <= self.capture_threshold

    def get_observation(self) -> tuple:
        o = self.model.get_observation("prey")
        prey = o["prey"]
        goal_reached = self.is_goal_reached(prey.location)
        if goal_reached:
            self.complete = True
        if self.has_predator:
            if o["predator"]:
                predator = o["predator"]
                captured = self.is_prey_captured(prey.location, predator.location)
                if captured:
                    self.complete = True
                return prey.location, \
                       prey.theta, \
                       self.goal_location, \
                       predator.location, \
                       predator.theta, \
                       captured, \
                       goal_reached
            else:
                return prey.location, \
                       prey.theta, \
                       self.goal_location, \
                       None, \
                       None, \
                       False, \
                       goal_reached
        else:
            return prey.location, prey.theta, self.goal_location, goal_reached

    def is_complete(self):
        return self.complete

    def set_action(self, speed: float, turning: float) -> None:
        action = AgentAction(speed, turning)
        self.model.set_agent_action("prey", action)

    def run(self) -> None:
        self.start()
        self.model.run()

    def start(self) -> None:
        self.complete = False
        self.model.set_agent_position("prey", self.start_location, math.pi / 2)
        if self.has_predator:
            predator_location = random.choice(self.spawn_locations)
            predator_theta = math.pi * 2 * random.random()
            self.model.set_agent_position("predator", predator_location, predator_theta)

    def step(self, action):
        done, truncated = False, False
        speed, turning = action[0], action[1]
        self.set_action(speed, turning)
        self.model.step()
        location, theta, goal_location, goal_reached = self.get_observation()
        obs = np.array([location.x, location.y, theta, action[0], action[1], goal_location.x, goal_location.y], dtype=np.float32)
        dx = abs(obs[0] - 1)
        dy = abs(obs[1] - 0.5)
        d = math.sqrt(dx ** 2 + dy ** 2)
        # d_reward = -d
        if self.is_goal_reached(location):
            reward = 100
            done = True
        else:
            reward = -d
        info = {"is success": done}
        self.current_step += 1
        if self.current_step > self.max_step:
            truncated = True
        self.current_episode_reward += reward
        if done or truncated:
            self.episode_reward_history.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return obs, reward, done, truncated, info

    def reset(self, seed = None):
        self.start()
        location, theta, goal_location, goal_reached = self.get_observation()
        obs = np.array([location.x, location.y, theta, 0.0, 0.0, goal_location.x, goal_location.y], dtype=np.float32)
        self.current_step = 1
        self.stop()
        return obs, {}

    def stop(self) -> None:
        self.model.stop()

    def show(self) -> None:
        if self.predator:
            if self.predator.destination:
                self.predator_destination.set(center=(self.predator.destination.x, self.predator.destination.y), color="b")
            if self.predator.destination_cell:
                self.predator_destination_cell.set(center=(self.predator.destination_cell.location.x, self.predator.destination_cell.location.y), color="g")
            self.predator_capture_area.set(center=(self.model.agents_data["predator"].location.x, self.model.agents_data["predator"].location.y), color="r")
        self.model.show()

    def seed(self, seed=123):
        self.np_random = np.random.default_rng(seed)
