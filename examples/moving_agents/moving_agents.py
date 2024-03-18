"""Main file for the moving agents simulation."""

import argparse
from dataclasses import dataclass
import random
import math
import multiprocessing

import numpy as np

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


@dataclass
class Room:
    """Class for the room where the agents move."""

    width: int
    height: int
    neibourhood_size: int
    number_of_agents: int
    total_time: int

    @staticmethod
    def levy_walk(agent, angle, step):
        """Move the agent in a Levy walk."""

        agent["x"] += step * math.cos(angle)
        agent["y"] += step * math.sin(angle)

    @staticmethod
    def random_walk(agent, angle, step):
        """Move the agent in a random walk."""

        if angle < math.pi / 2:
            agent["x"] += step
        elif angle < math.pi:
            agent["y"] += step
        elif angle < 3 * math.pi / 2:
            agent["x"] -= step
        else:
            agent["y"] -= step

    @staticmethod
    def brownian_motion(agent, angle, step):
        """Move the agent in a Brownian motion."""

        agent["x"] += step * math.cos(angle)
        agent["y"] += step * math.sin(angle)

    def evolve(self, mode, movement_scale_factor):
        """Evolve the agents in the room."""

        agents = []
        for agent_idx in range(self.number_of_agents):
            agent = {
                "x": random.randint(0, self.width),
                "y": random.randint(0, self.height),
                "angle": random.uniform(0, 2 * math.pi),
                "color": BLUE if agent_idx == 0 else GREEN,
            }
            agents.append(agent)

        if mode == 0:
            step_length = movement_scale_factor
            steps = [step_length] * self.total_time * self.number_of_agents
        elif mode == 1:
            beta = movement_scale_factor / (movement_scale_factor - 1)
            steps = np.random.pareto(beta, self.total_time * self.number_of_agents)
        else:
            steps = np.random.normal(
                0,
                movement_scale_factor * np.sqrt(np.pi / 2),
                self.total_time * self.number_of_agents,
            )

        angles = np.random.uniform(
            0, 2 * math.pi, self.total_time * self.number_of_agents
        )

        for time_idx in range(self.total_time):

            for agent_idx, agent in enumerate(agents):

                index = time_idx * self.number_of_agents + agent_idx
                # walk
                if mode == 0:
                    self.levy_walk(agent, angles[index], steps[index])
                elif mode == 1:
                    self.random_walk(agent, angles[index], steps[index])
                else:
                    self.brownian_motion(agent, angles[index], steps[index])

                # Wrap agents around the screen
                agent["x"] = agent["x"] % self.width
                agent["y"] = agent["y"] % self.height

                # Draw agents

                if agent_idx == 0:
                    pass

                else:

                    if (agent["x"] - agents[0]["x"]) ** 2 + (
                        agent["y"] - agents[0]["y"]
                    ) ** 2 <= self.neibourhood_size**2:
                        if random.random() < 0.01:
                            agent["color"] = RED

        return len([agent for agent in agents if agent["color"] == RED])


def single_run(
    mode_choice,
    movement_scale_factor,
    num_agents,
    time,
    room_width,
    room_height,
    neighbourhood_size,
):
    """Run the simulation once."""

    # pylint: disable=too-many-arguments

    room = Room(room_width, room_height, neighbourhood_size, num_agents, time)

    result = room.evolve(mode_choice, movement_scale_factor)
    return result


def multi_run(
    mode_choice,
    movement_scale_factor,
    num_agents,
    time,
    room_width,
    room_height,
    neighbourhood_size,
    num_runs,
    seed=None,
):
    """Run the simulation multiple times."""

    # pylint: disable=too-many-arguments

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    model_args = [
        (
            mode_choice,
            movement_scale_factor,
            num_agents,
            time,
            room_width,
            room_height,
            neighbourhood_size,
        )
    ] * num_runs

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(single_run, model_args)

    avg_result = sum(results) / num_runs
    # std_dev = np.std(results)

    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(str(avg_result))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        help="0: Levy walk, 1: Random walk, 2: Brownian motion",
    )

    parser.add_argument(
        "--movement_scale_factor",
        type=float,
        default=3,
        help="Scale factor for the movement",
    )

    parser.add_argument(
        "--num_agents",
        type=int,
        default=100,
        help="Number of agents",
    )

    parser.add_argument(
        "--time",
        type=int,
        default=6000,
        help="Total time",
    )

    parser.add_argument(
        "--room_width",
        type=int,
        default=1000,
        help="Room width",
    )

    parser.add_argument(
        "--room_height",
        type=int,
        default=1000,
        help="Room height",
    )

    parser.add_argument(
        "--neighbourhood_size",
        type=int,
        default=50,
        help="Neighbourhood size",
    )

    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    args = parser.parse_args()

    multi_run(
        args.mode,
        args.movement_scale_factor,
        args.num_agents,
        args.time,
        args.room_width,
        args.room_height,
        args.neighbourhood_size,
        args.num_runs,
        args.seed,
    )
