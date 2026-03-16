from abc import ABC, abstractmethod

from medagent.schema import OrchestratorState


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, state: OrchestratorState) -> str:
        raise NotImplementedError

