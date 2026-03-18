from abc import ABC, abstractmethod

from medagent.schema import OrchestratorState


class BaseAgent(ABC):
    task_name: str = "base"
    display_name: str = "Base-Agent"

    @abstractmethod
    def run(self, state: OrchestratorState) -> str:
        raise NotImplementedError

    def _active_prompt(self, state: OrchestratorState) -> str:
        return str(state.artifacts.get("active_prompt", "")).strip()

    def _llm_runtime(self, state: OrchestratorState):
        return state.artifacts.get("llm_runtime")

    def _user_query(self, state: OrchestratorState) -> str:
        return state.messages[-1].content if state.messages else ""

    def _generate(
        self,
        state: OrchestratorState,
        user_prompt: str,
        *,
        image_path: str | None = None,
        max_new_tokens: int = 384,
    ) -> str | None:
        runtime = self._llm_runtime(state)
        system_prompt = self._active_prompt(state)
        if runtime is None or not system_prompt:
            return None
        try:
            try:
                from langchain_core.prompts import ChatPromptTemplate

                template = ChatPromptTemplate.from_messages(
                    [
                        ("system", "{system_prompt}"),
                        ("human", "{user_prompt}"),
                    ]
                )
                formatted = template.format_messages(system_prompt=system_prompt, user_prompt=user_prompt)
                rendered_system = formatted[0].content if len(formatted) > 0 else system_prompt
                rendered_user = formatted[1].content if len(formatted) > 1 else user_prompt
            except Exception:
                rendered_system = system_prompt
                rendered_user = user_prompt

            return runtime.chat(
                system_prompt=rendered_system,
                user_prompt=rendered_user,
                image_path=image_path,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback path
            state.artifacts.setdefault("runtime_errors", []).append(
                {"agent": self.display_name, "error": str(exc)}
            )
            return None
