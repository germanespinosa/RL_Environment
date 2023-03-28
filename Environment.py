import math

from Predator import Predator
from Agent import *
from Model import Model
from Paths import *
from cellworld import *
import random


class Environment:
    def __init__(self, world_name: str, freq: int = 100, has_predator: bool = False):
        self.world = World.get_from_parameters_names("hexagonal", "canonical", world_name)
        self.model = Model(self.world, freq)
        self.goal_location = Location(1, .5)
        self.start_location = Location(0, .5)
        self.has_predator = has_predator
        self.goal_threshold = self.world.implementation.cell_transformation.size / 2
        self.capture_threshold = self.world.implementation.cell_transformation.size
        self.model.add_agent("prey", None, self.start_location, math.pi/2, "b", pauto_update=False)
        self.goal_area = self.model.display.circle(location=self.goal_location,
                                                   color="g",
                                                   alpha=.5,
                                                   radius=self.goal_threshold)
        self.complete=False
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
            predator_location = random.choice(self.spawn_locations)
            predator_theta = math.pi * 2 * random.random()
            self.model.add_agent("predator", self.predator, predator_location, predator_theta, "r")
            self.predator_destination = self.model.display.circle(location=predator_location, color="b", alpha=.5, radius=0.01)
            self.predator_destination_cell = self.model.display.circle(location=predator_location, color="g", alpha=.5, radius=0.02)
            self.predator_capture_area = self.model.display.circle(location=predator_location, color="r", alpha=.5, radius=self.capture_threshold)

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
        self.model.run()

    def stop(self) -> None:
        self.model.stop()

    def show(self) -> None:
        if self.predator.destination:
            self.predator_destination.set(center=(self.predator.destination.x, self.predator.destination.y), color="b")
        if self.predator.destination_cell:
            self.predator_destination_cell.set(center=(self.predator.destination_cell.location.x, self.predator.destination_cell.location.y), color="g")
        self.predator_capture_area.set(center=(self.model.agents_data["predator"].location.x, self.model.agents_data["predator"].location.y), color="r")
        self.model.show()
