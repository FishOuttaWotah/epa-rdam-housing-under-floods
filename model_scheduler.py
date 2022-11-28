from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Sequence, Tuple, Union, Type, Mapping
import random
from collections import defaultdict
from collections import OrderedDict  # note similar name for class from typing (mypy)
# from mesa import Model, Agent
import mesa
from enum import Enum
# if TYPE_CHECKING:
import model_init
import agent_base

"""
_____________________________________________________

Model Scheduler, modified from Mesa
_____________________________________________________
"""
"""
A rewrite of the Scheduler class used in MESA, due to the high amount of extensions needed for this model over the Scheduler variants offered by MESA 
- staged activation by type (already seen in Taberna)
- dynamic activation across runs (ie. changing type like seller/buyer back into normal agents)
* includes the AGENT_PRIORITY dictionary, describing the priority of agents and the order of their staging
(oh no what about actions that 'wait' on other agents? probably not required here?)
* the _agents object works slightly differently to MESA's implementation
* maybe have the scheduler and ledger be embedded in the same thing?? 

"""

ModelAgents = agent_base.CustomAgent


# Useful: method_list = [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]


# TODO: think about if the ledger management should be incorporated here

# set up Scheduler:
class Scheduler:  # probably will change the name to more specific
    # includes flexible scheduling, shuffling, staged scheduling, and queued scheduling
    # NB: define the keywords mentioned in ^
    # dict types for different kinds of agents (household,
    def __init__(self,
                 model: model_init.RHuCC_Model,
                 timestep: int,
                 agent_priority,  # typing.OrderedDict[Union[ModelAgents, str], Sequence[str]],
                 shuffle: bool = False,  # probably should set a dict for different classes of agents
                 shuffle_between_stages=False):
        # inputs: types of agents (household {buyer, seller}, firm {capital, goods, consumption}, Background)
        self.MODEL = model
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages

        # agent priority serves two functions: one for the agent types to follow, and their internal methods
        self.AGENT_PRIORITY: Sequence[Union[Tuple]] = agent_priority
        # self.agent_specials = {}  # mostly used for buyer/seller relations, but could be expanded
        # contains lists of tuples, 0 = agent class, 1 = agent step list.
        # if tuple list is >2, assume to be with an alias on index 2
        # for idx, agent_input in enumerate(self.AGENT_PRIORITY):
        #     if len(agent_input) != 2:
        #         # input custom alias to custom dict
        #         self.agent_specials[agent_input[2]] = []

        # validate agent stages may not work with str...
        # Check to see if the agent's stated stages are actually available in the class object
        agent_base.validate_agent_stages(self.AGENT_PRIORITY)
        # agentlabels should be the agent class
        self.steps: int = 0
        self.time: int = 0  # representation in months, can be converted to years
        self.TIMESTEP: int = timestep

        # create the dict for the model agents. Might need to discriminate based on agent type
        # Tiered Dict
        self.agents_by_type: defaultdict[str, Dict[str, ModelAgents]] = defaultdict(
            dict)  # NB: usage of normal dict instead of OrderedDict in default MESA
        self.agents: Dict = defaultdict(dict)
        # ^ usage of normal dict instead of OrderedDict/defaultdict as default

    def add(self, agent: ModelAgents, overwrite: bool = False) -> None:
        if agent.unique_id in self.agents and not overwrite:
            raise Exception(
                f"Agent with unique id {repr(agent.unique_id)} already added to scheduler"
            )

        self.agents_by_type[agent.__class__.__name__][agent.unique_id] = agent
        self.agents[agent.unique_id] = agent
        # bonus: could add function for bulk addition?

    def remove(self, agent: ModelAgents) -> None:
        del self.agents[agent.unique_id]
        del self.agents_by_type[agent.__class__.__name__][agent.unique_id]

    def step(self) -> None:
        # conduct time-advance (if anything happens, it happened in 'this timestep')
        self.time += self.TIMESTEP
        self.steps += 1

        # need to iterate across different classes of agents
        for agent_type, stage_list in self.AGENT_PRIORITY:

            # if len(instructions) > 2:
            #     # use the specials
            #     special_key = instructions[2]  # get special alias
                # agent_keys: list = self.agent_specials[special_key]  # retrieve special list (buyer/seller)
            # else:
            # get all agents for this class
            agent_keys = list(self.agents_by_type[agent_type.__name__].keys())

            if self.shuffle:
                random.shuffle(agent_keys)  # note shuffling efficiency is O(n) (linear with size of list)

            for stage in stage_list:
                if self.shuffle_between_stages:
                    random.shuffle(agent_keys)
                for idx in agent_keys:
                    getattr(self.agents[idx], stage)()  # run stage

    def year_convert(self):
        return self.time // 12, self.time % 12


def extract_custom_entries(input_dict: Mapping, custom_type=str):
    # used to extract dict keys that are of unique type (ie. normal is class object, unique is str)
    key_list = []
    for key in input_dict.keys():
        if type(key) == custom_type:
            key_list.append(key)

    return key_list


## includes background agent/scheduler here? Or merge as part of scheduler?
## includes stochastic agent decider and buyer agent generator connections

## Need to generate data entities in the Model for the ledgers

class DummyAgent1(agent_base.CustomAgent):
    stagelist = ('stage1', 'stage2', 'stage3')

    def __init__(self, unique_id, model=None):
        super(DummyAgent1, self).__init__(unique_id, model)
        self.stage_history = []

    def stage1(self):
        print('I am stage 1')
        self.stage_history.append(1)

    def stage2(self):
        print('I am stage 2')
        self.stage_history.append(2)

    def stage3(self):
        print('I am stage 3')
        self.stage_history.append(3)


class DummyAgent2(DummyAgent1):
    stagelist = ('stage3', 'stage2', 'stage1')

    def __init__(self, unique_id, model=None):
        super(DummyAgent2, self).__init__(unique_id, model)


if __name__ == '__main__':
    priority = {DummyAgent2: DummyAgent2.stagelist, DummyAgent1: DummyAgent1.stagelist}
    # priority = [(DummyAgent2, DummyAgent2.stagelist), (DummyAgent1, DummyAgent1.stagelist)]
    scheduley = Scheduler(model=None, timestep=1, agent_priority=priority)

    agent1 = DummyAgent1(unique_id='1')
    agent2 = DummyAgent2(unique_id='2')

    scheduley.add(agent1)
    scheduley.add(agent2)

    scheduley.step()
