from __future__ import annotations
from typing import TYPE_CHECKING, Mapping, Any, Sequence

if TYPE_CHECKING:
    import model_init


class CustomAgent:
    # dummy implementation similar to mesa.Agent, but with the distinction of having a stagelist and str-based unique-id
    stagelist: Sequence[str] = ()  # empty

    def __init__(self, model: model_init.RHuCC_Model, unique_id: str= None):
        self.unique_id = unique_id
        self.model = model


class DelayAssigned:  # create empty class
    pass

def validate_agent_stages(agent_priority):  # TODO: change Any typing to concrete agent class
    """
    Preliminary check to see if
    1) the agent's stagelist-attribute exists,
    2) the stated stages in the stagelist exists

    Does not check whether the stages run correctly or not!

    :param agent_priority: dictionary with the key as the agent class (eg. Firm), and an arraylike listing the stages the agent must go through.
    :return:
    """
    diagnosis = []
    for instructions in agent_priority:
        agent_type = instructions[0]
        stagelist = instructions[1]
        for stage in stagelist:
            if not hasattr(agent_type, stage):
                diagnosis.append(f"{agent_type.__name__}:{stage}")  # only save the human-readable name of class

    if bool(diagnosis):  # non-empty list would trigger
        raise ValueError(f'Agent types with missing stages (agent_type, stage):\n{diagnosis}')
    # else:
    # print('Agent stages validated')

class Buyer:
    def init(self):
        # define the rules of the buyer here
        pass