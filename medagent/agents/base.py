from abc import ABC, abstractmethod

from medagent.schema import OrchestratorState


class BaseAgent(ABC):
    task_name: str = "base"
    display_name: str = "Base-Agent"

    @abstractmethod
    def run(self, state: OrchestratorState) -> str:
        raise NotImplementedError
