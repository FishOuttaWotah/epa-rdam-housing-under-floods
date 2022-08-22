"""
scheduler;
- staged activation by type (already seen in Taberna)
- dynamic activation across runs (ie. changing type like seller/buyer back into normal agents)
"""

from typing import TYPE_CHECKING, Dict, Any, Sequence, Tuple, Union
import typing
from collections import OrderedDict # note similar name for class from typing (mypy)
# from mesa import Model, Agent
import mesa
from enum import Enum

"""
_____________________________________________________

MESA Model Scheduler, modified
_____________________________________________________
"""
"""
A rewrite of the Scheduler class used in MESA, due to the high amount of extensions needed for this model over the Scheduler variants offered by MESA 

* the _agents object works slightly differently to MESA's implementation

"""
# TODO: create Union type of agents for the model
ModelAgents = Any

# set up enum keys

# Useful: method_list = [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

## agent types that should be checked for here
AgentLabels = str

#TODO: think about if the ledger management should be incorporated here

# set up Scheduler:
class Scheduler:  # probably will change the name to more specific
    # includes flexible scheduling, shuffling, staged scheduling, and queued scheduling
    # NB: define the keywords mentioned in ^
    # dict types for different kinds of agents (household,
    def __init__(self,
                 model: mesa.Model,
                 timestep:int,
                 agent_priority:Sequence[Tuple[AgentLabels, Sequence[str]]]):
        # inputs: types of agents (household {buyer, seller}, firm {capital, goods, consumption}, Background)
        self.MODEL = model

        # agent priority serves two functions: one for the agent types to follow, and their internal methods
        # convert agent_priority input into OrderedDict
        self.AGENT_PRIORITY: typing.OrderedDict[AgentLabels, Sequence[str]] = OrderedDict(agent_priority) # could include the agent type priority and stages as tuples? (
        self.steps = 0
        self.time: int = 0  # representation in months, can be converted to years
        self.TIMESTEP: int = timestep

        # create the dict for the model agents. Might need to discriminate based on agent type
        # Tiered  Dict
        self._agents: Dict[str, Dict[str, ModelAgents]] = dict((agent_type, {}) for agent_type in self.AGENT_PRIORITY.keys())  # NB: usage of normal dict instead of OrderedDict

        #

    def add(self,agent: ModelAgents) -> None:
        if agent.unique_id in self._agents:
            raise Exception(
                f"Agent with unique id {repr(agent.unique_id)} already added to scheduler"
            )
        self._agents[agent.unique_id] = agent

    def remove(self, agent: ModelAgents) -> None:
        del self._agents[agent.unique_id]

    def step(self) -> None:
        # need to iterate across different classes of agents


        pass

    def get_agent_count(self):
        return

    def year_convert(self):
        return self.time // 12, self.time % 12
## includes background agent/scheduler here? Or merge as part of scheduler?
## includes stochastic agent decider and buyer agent generator connections

## Need to generate data entities in the Model for the ledgers