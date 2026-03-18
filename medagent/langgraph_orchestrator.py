from __future__ import annotations

import time
from typing import Any, TypedDict

from medagent.orchestrator import Orchestrator
from medagent.schema import AgentMessage, OrchestratorState, VisitRecord
from medagent.services.clinical_pathway import extract_profile_updates
from medagent.services.multiturn import (
    build_preliminary_assessment,
    new_visit_id,
    refresh_visit_record,
    select_followup_questions,
    serialize_visit_record,
    should_stop_visit,
    update_state_from_user_message,
)


class WorkflowState(TypedDict, total=False):
    mode: str
    user_id: str
    user_text: str
    age: int | None
    sex: str | None
    image_path: str | None
    visit_id: str
    existing_visit: dict[str, Any] | None
    core_state: OrchestratorState
    sections: list[str]
    turn_outputs: dict[str, str]
    final_response: str
    follow_up_questions: list[str]
    visit_completed: bool
    turn_result: dict[str, Any]


class LangChainOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._graph = None
        self._turn_graph = None

    def _build_bundle_query(self, core_state: OrchestratorState, latest_user_text: str) -> str:
        recent_user_turns = [msg.content for msg in core_state.messages if msg.role == "user"][-4:]
        query_parts = recent_user_turns or [latest_user_text]
        if core_state.visit_record and core_state.visit_record.human_readable_summary:
            query_parts.append(core_state.visit_record.human_readable_summary)
        return "\n".join(part for part in query_parts if part)

    def _prepare_state(self, state: WorkflowState) -> WorkflowState:
        user_id = state["user_id"]
        user_text = state["user_text"]
        age = state.get("age")
        sex = state.get("sex")
        image_path = state.get("image_path")

        ctx = self._build_user_context(user_id=user_id, age=age, sex=sex)
        core_state = OrchestratorState(
            user_context=ctx,
            messages=[AgentMessage(role="user", content=user_text)],
        )
        route = self.intent_router.route(user_text)
        core_state.intent = route.intent
        core_state.artifacts["route"] = {"intent": route.intent, "reason": route.reason}
        core_state.artifacts["prompt_profiles"] = {
            "orchestrator": self._load_prompt("orchestrator"),
            "memory": self._load_prompt("memory"),
        }
        core_state.artifacts["llm_runtime"] = self.llm_runtime
        core_state.artifacts["image_path"] = image_path
        core_state.risk_level = self.safety.detect_risk(user_text)
        core_state.tasks = self._plan(route.intent)

        self.memory.append_turn(user_id, user_text)
        if ctx.age is not None:
            self.memory.upsert_profile_fact(user_id, "age", str(ctx.age))
        if ctx.sex is not None:
            self.memory.upsert_profile_fact(user_id, "sex", ctx.sex)

        bundle = self.rag.build_bundle(
            user_id=user_id,
            query=user_text,
            memory_store=self.memory,
            knowledge_top_k=3,
            memory_top_k=3,
        )
        core_state.artifacts["rag"] = self._serialize_rag_bundle(bundle)
        core_state.artifacts["knowledge_docs"] = core_state.artifacts["rag"]["knowledge_docs"]
        core_state.artifacts["education_docs"] = core_state.artifacts["rag"]["knowledge_docs"]
        core_state.artifacts["memory_hits"] = core_state.artifacts["rag"]["memory_hits"]
        core_state.artifacts["bundle"] = bundle

        return {
            **state,
            "mode": "single_shot",
            "core_state": core_state,
            "sections": [],
        }

    def _prepare_turn_state(self, state: WorkflowState) -> WorkflowState:
        user_id = state["user_id"]
        user_text = state["user_text"]
        age = state.get("age")
        sex = state.get("sex")
        image_path = state.get("image_path")
        existing_visit = state.get("existing_visit")

        if existing_visit:
            core_state = existing_visit["core_state"]
            core_state.user_context = self._build_user_context(user_id=user_id, age=age, sex=sex)
            core_state.turn_index += 1
            core_state.final_response = ""
            core_state.stop_reason = ""
            core_state.messages.append(AgentMessage(role="user", content=user_text))
        else:
            visit_id = state.get("visit_id") or new_visit_id(user_id)
            ctx = self._build_user_context(user_id=user_id, age=age, sex=sex)
            core_state = OrchestratorState(
                user_context=ctx,
                messages=[AgentMessage(role="user", content=user_text)],
                visit_record=VisitRecord(visit_id=visit_id, user_id=user_id),
            )

        route = self.intent_router.route(user_text)
        if not existing_visit or route.intent.startswith("report") or core_state.intent == "patient_education":
            core_state.intent = route.intent
        core_state.artifacts["route"] = {"intent": route.intent, "reason": route.reason}
        core_state.artifacts["prompt_profiles"] = {
            "orchestrator": self._load_prompt("orchestrator"),
            "memory": self._load_prompt("memory"),
        }
        core_state.artifacts["llm_runtime"] = self.llm_runtime
        core_state.artifacts["image_path"] = image_path
        core_state.risk_level = self.safety.detect_risk(user_text)
        core_state.tasks = self._plan(core_state.intent)

        self.memory.append_turn(user_id, user_text)
        if core_state.user_context.age is not None:
            self.memory.upsert_profile_fact(user_id, "age", str(core_state.user_context.age))
        if core_state.user_context.sex is not None:
            self.memory.upsert_profile_fact(user_id, "sex", core_state.user_context.sex)

        update_state_from_user_message(core_state)
        bundle_query = self._build_bundle_query(core_state, user_text)
        bundle = self.rag.build_bundle(
            user_id=user_id,
            query=bundle_query,
            memory_store=self.memory,
            knowledge_top_k=3,
            memory_top_k=4,
        )
        core_state.artifacts["rag"] = self._serialize_rag_bundle(bundle)
        core_state.artifacts["knowledge_docs"] = core_state.artifacts["rag"]["knowledge_docs"]
        core_state.artifacts["education_docs"] = core_state.artifacts["rag"]["knowledge_docs"]
        core_state.artifacts["memory_hits"] = core_state.artifacts["rag"]["memory_hits"]
        core_state.artifacts["bundle"] = bundle

        visit_id = core_state.visit_record.visit_id if core_state.visit_record else state.get("visit_id", "")
        return {
            **state,
            "mode": "visit_turn",
            "visit_id": visit_id,
            "core_state": core_state,
            "turn_outputs": {},
            "follow_up_questions": [],
            "visit_completed": False,
        }

    def _route_branch(self, state: WorkflowState) -> str:
        intent = state["core_state"].intent
        if intent == "patient_education":
            return "flow_education"
        if intent == "education_report":
            return "flow_report_then_education"
        if intent == "report_followup":
            return "flow_report_then_intake"
        if intent == "medication_followup":
            return "flow_intake_then_medication"
        return "flow_general_intake"

    def _store_output(self, state: WorkflowState, task: str, content: str) -> WorkflowState:
        if state.get("mode") == "visit_turn":
            outputs = dict(state.get("turn_outputs", {}))
            outputs[task] = content
            return {**state, "turn_outputs": outputs}
        sections = list(state.get("sections", []))
        sections.append(f"[{self._section_title(task)}]\n{content}")
        return {**state, "sections": sections}

    def _task_intake(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        core_state.artifacts["active_prompt"] = self._load_prompt("intake")
        core_state.artifacts["prompt_profiles"]["intake"] = core_state.artifacts["active_prompt"]
        return self._store_output(state, "intake", self.intake.run(core_state))

    def _task_triage(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        core_state.artifacts["active_prompt"] = self._load_prompt("triage")
        core_state.artifacts["prompt_profiles"]["triage"] = core_state.artifacts["active_prompt"]
        return self._store_output(state, "triage", self.triage.run(core_state))

    def _task_report(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        core_state.artifacts["active_prompt"] = self._load_prompt("report")
        core_state.artifacts["prompt_profiles"]["report"] = core_state.artifacts["active_prompt"]
        return self._store_output(state, "report", self.report.run(core_state))

    def _task_education(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        core_state.artifacts["active_prompt"] = self._load_prompt("education")
        core_state.artifacts["prompt_profiles"]["education"] = core_state.artifacts["active_prompt"]
        return self._store_output(state, "education", self.education.run(core_state))

    def _task_medication(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        core_state.artifacts["active_prompt"] = self._load_prompt("medication")
        core_state.artifacts["prompt_profiles"]["medication"] = core_state.artifacts["active_prompt"]
        return self._store_output(state, "medication", self.medication.run(core_state))

    def _task_rag(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        bundle = core_state.artifacts["bundle"]
        core_state.evidence = [{"source": doc.source, "chunk": doc.chunk} for doc in bundle.knowledge_docs]
        return self._store_output(state, "rag_summary", self._format_rag_summary(bundle))

    def _after_report(self, state: WorkflowState) -> str:
        if state["core_state"].intent == "education_report":
            return "education"
        return "intake"

    def _after_intake(self, state: WorkflowState) -> str:
        if state["core_state"].intent == "medication_followup":
            return "medication"
        return "triage"

    def _after_triage(self, state: WorkflowState) -> str:
        return "medication"

    def _after_medication(self, state: WorkflowState) -> str:
        if state["core_state"].intent == "medication_followup":
            return "triage_after_medication"
        return "rag_summary"

    def _fallback_turn_response(
        self,
        state: OrchestratorState,
        follow_up_questions: list[str],
        visit_completed: bool,
    ) -> str:
        triage = state.artifacts.get("triage", {})
        intake = state.artifacts.get("intake", {})
        lines: list[str] = []

        if triage.get("level") in {"emergency", "consider_admission"}:
            lines.append(f"先提醒：{triage.get('label', '建议立即线下就医')}")

        if follow_up_questions and not visit_completed:
            lines.append("本轮追问：")
            for question in follow_up_questions[:2]:
                lines.append(f"- {question}")
            return "\n".join(lines)

        lines.append(f"初步判断：{build_preliminary_assessment(state)}")
        if triage.get("label"):
            lines.append(f"分诊建议：{triage['label']}，建议前往{triage.get('department', '相应专科')}。")
        tests = intake.get("recommended_tests", [])
        if tests:
            lines.append(f"建议检查：{'、'.join(tests[:4])}")
        return "\n".join(lines)

    def _format_completed_turn_response(self, state: OrchestratorState) -> str:
        triage = state.artifacts.get("triage", {})
        intake = state.artifacts.get("intake", {})
        label = triage.get("label") or (state.visit_record.triage_label if state.visit_record else "建议线下就医")
        department = triage.get("department", "相应专科")
        assessment = build_preliminary_assessment(state)
        tests = intake.get("recommended_tests", [])[:4]

        lines = [str(label)]
        if assessment:
            lines.append(assessment)
        if department:
            lines.append(f"建议尽快前往{department}完成线下评估。")
        if tests:
            lines.append(f"建议优先做这些检查：{'、'.join(tests)}")
        return "\n".join(lines)

    def _compose_turn_response(
        self,
        state: OrchestratorState,
        turn_outputs: dict[str, str],
        follow_up_questions: list[str],
        visit_completed: bool,
    ) -> str:
        if visit_completed and state.stop_reason in {
            "patient_accepted_disposition",
            "user_ended_conversation",
            "question_queue_exhausted",
            "enough_information_collected",
            "max_turns_reached",
        }:
            return self._format_completed_turn_response(state)

        system_prompt = self._load_prompt("orchestrator")
        if not system_prompt:
            return self._fallback_turn_response(state, follow_up_questions, visit_completed)

        triage = state.artifacts.get("triage", {})
        intake = state.artifacts.get("intake", {})
        user_text = state.messages[-1].content if state.messages else ""
        followup_text = "\n".join(f"- {question}" for question in follow_up_questions[:2]) or "无"
        stop_label = state.stop_reason or "continue"
        user_prompt = "\n".join(
            [
                f"当前用户输入：{user_text}",
                f"当前轮次：第 {state.turn_index} 轮",
                f"是否结束本次首程：{'是' if visit_completed else '否'}",
                f"结束原因：{stop_label}",
                f"当前分诊：{triage.get('label', '待补充更多信息')}",
                f"建议科室：{triage.get('department', '待定')}",
                f"红旗信号：{'、'.join(state.red_flags) if state.red_flags else '暂无'}",
                f"建议追问：\n{followup_text}",
                f"门诊专家候选：{turn_outputs.get('intake', '')}",
                f"紧急分诊候选：{turn_outputs.get('triage', '')}",
                f"用药医师候选：{turn_outputs.get('medication', '')}",
                f"影像医师候选：{turn_outputs.get('report', '')}",
                f"健康顾问候选：{turn_outputs.get('education', '')}",
                f"建议检查：{'、'.join(intake.get('recommended_tests', [])[:4]) or '暂无'}",
                "请输出给患者看的最终回复，要求：",
                "1. 不要出现 agent 名称。",
                "2. 如果需要尽快就诊，把就医提示放在第一句。",
                "3. 如果还需要继续追问，只问 1 到 2 个问题，每行一个，不带编号，不写“先确认两个最关键的问题”之类的话。",
                "4. 如果已经可以收束，给出初步判断、分诊建议和具体检查建议。",
                "5. 整体要简洁、口语化，不要像百科文章。",
            ]
        )
        try:
            response = self.llm_runtime.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=state.artifacts.get("image_path"),
                max_new_tokens=320,
            )
            if response:
                return response
        except Exception:
            pass
        return self._fallback_turn_response(state, follow_up_questions, visit_completed)

    def _compose_visit_turn(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        record = refresh_visit_record(core_state)
        visit_completed, stop_reason = should_stop_visit(core_state, max_turns=self.config.max_turns)

        follow_up_candidates = []
        if not visit_completed:
            follow_up_candidates = select_followup_questions(core_state, limit=2)
            if not follow_up_candidates:
                visit_completed = True
                stop_reason = "question_queue_exhausted"

        core_state.stop_reason = stop_reason
        record = refresh_visit_record(core_state)
        follow_up_questions = [item.question for item in follow_up_candidates]
        response = self._compose_turn_response(
            core_state,
            state.get("turn_outputs", {}),
            follow_up_questions,
            visit_completed,
        )

        turn_result = {
            "visit_id": record.visit_id,
            "response": response,
            "visit_completed": visit_completed,
            "stop_reason": stop_reason,
            "follow_up_questions": follow_up_questions,
            "triage_label": record.triage_label,
            "recommended_tests": list(record.recommended_tests),
            "preliminary_assessment": record.preliminary_assessment,
            "visit_record": serialize_visit_record(record),
        }
        core_state.final_response = response
        return {
            **state,
            "follow_up_questions": follow_up_questions,
            "visit_completed": visit_completed,
            "turn_result": turn_result,
            "final_response": response,
        }

    def _persist_visit_turn(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        user_id = state["user_id"]
        visit_id = state["visit_id"]
        turn_result = state["turn_result"]

        core_state.messages.append(AgentMessage(role="assistant", content=turn_result["response"]))

        profile_updates = extract_profile_updates(core_state)
        for allergy in profile_updates.get("allergies", []):
            self.memory.append_profile_item(user_id, "allergies", allergy)
        for condition in profile_updates.get("chronic_history", []):
            self.memory.append_profile_item(user_id, "chronic_history", condition)
        for med in profile_updates.get("current_meds", []):
            self.memory.append_profile_item(user_id, "current_meds", med)
        if profile_updates.get("pregnancy_status"):
            self.memory.upsert_profile_fact(user_id, "pregnancy_status", profile_updates["pregnancy_status"])

        if core_state.visit_record:
            self.memory.upsert_profile_fact(user_id, "recent_assessment", core_state.visit_record.triage_label or "待进一步评估")

        report_summary = core_state.artifacts.get("report_parsed", {}).get("summary")
        if report_summary:
            self.memory.append_source_document(
                user_id,
                {
                    "title": "外来报告摘要",
                    "summary": report_summary,
                    "visit_id": visit_id,
                },
            )

        if state["visit_completed"] and core_state.visit_record:
            serialized = serialize_visit_record(core_state.visit_record)
            self.memory.append_visit_record(user_id, serialized)
            self.memory.append_episode(user_id, topic="visit_summary", content=core_state.visit_record.human_readable_summary)
            self.memory.remove_active_visit(user_id, visit_id)
        else:
            self.memory.save_active_visit(
                user_id,
                visit_id,
                {
                    "core_state": core_state,
                },
            )
        return state

    def _finalize(self, state: WorkflowState) -> WorkflowState:
        core_state = state["core_state"]
        draft = "\n".join(state.get("sections", []))
        if self.config.require_citations and core_state.evidence:
            refs = "；".join(item["source"] for item in core_state.evidence)
            draft += f"\n[引用] {refs}"

        profile_updates = extract_profile_updates(core_state)
        user_id = state["user_id"]
        for allergy in profile_updates.get("allergies", []):
            self.memory.append_profile_item(user_id, "allergies", allergy)
        for condition in profile_updates.get("chronic_history", []):
            self.memory.append_profile_item(user_id, "chronic_history", condition)
        for med in profile_updates.get("current_meds", []):
            self.memory.append_profile_item(user_id, "current_meds", med)
        if profile_updates.get("pregnancy_status"):
            self.memory.upsert_profile_fact(user_id, "pregnancy_status", profile_updates["pregnancy_status"])

        triage_label = core_state.artifacts.get("triage", {}).get("label", "未完成分诊")
        self.memory.upsert_profile_fact(user_id, "recent_assessment", triage_label)
        first_task = core_state.tasks[0] if core_state.tasks else "general"
        self.memory.append_episode(user_id, topic=first_task, content=draft[:500])

        if self.config.enable_memory_fusion:
            draft = self.memory_fusion.generate(user_id=user_id, query=state["user_text"], draft=draft)
        core_state.final_response = self.safety.enforce(draft, core_state.risk_level)
        return {**state, "final_response": core_state.final_response}

    def _build_graph(self):
        if self._graph is not None:
            return self._graph

        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(WorkflowState)
        graph.add_node("prepare", self._prepare_state)
        graph.add_node("report", self._task_report)
        graph.add_node("intake", self._task_intake)
        graph.add_node("triage", self._task_triage)
        graph.add_node("triage_after_medication", self._task_triage)
        graph.add_node("medication", self._task_medication)
        graph.add_node("education", self._task_education)
        graph.add_node("rag_summary", self._task_rag)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "prepare")
        graph.add_conditional_edges(
            "prepare",
            self._route_branch,
            {
                "flow_education": "education",
                "flow_report_then_education": "report",
                "flow_report_then_intake": "report",
                "flow_intake_then_medication": "intake",
                "flow_general_intake": "intake",
            },
        )
        graph.add_conditional_edges(
            "report",
            self._after_report,
            {
                "education": "education",
                "intake": "intake",
            },
        )
        graph.add_conditional_edges(
            "intake",
            self._after_intake,
            {
                "medication": "medication",
                "triage": "triage",
            },
        )
        graph.add_conditional_edges(
            "triage",
            self._after_triage,
            {
                "medication": "medication",
            },
        )
        graph.add_conditional_edges(
            "medication",
            self._after_medication,
            {
                "triage_after_medication": "triage_after_medication",
                "rag_summary": "rag_summary",
            },
        )
        graph.add_edge("triage_after_medication", "rag_summary")
        graph.add_edge("education", "rag_summary")
        graph.add_edge("rag_summary", "finalize")
        graph.add_edge("finalize", END)

        self._graph = graph.compile()
        return self._graph

    def _build_turn_graph(self):
        if self._turn_graph is not None:
            return self._turn_graph

        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(WorkflowState)
        graph.add_node("prepare_turn", self._prepare_turn_state)
        graph.add_node("report", self._task_report)
        graph.add_node("intake", self._task_intake)
        graph.add_node("triage", self._task_triage)
        graph.add_node("triage_after_medication", self._task_triage)
        graph.add_node("medication", self._task_medication)
        graph.add_node("education", self._task_education)
        graph.add_node("rag_summary", self._task_rag)
        graph.add_node("compose_turn", self._compose_visit_turn)
        graph.add_node("persist_turn", self._persist_visit_turn)

        graph.add_edge(START, "prepare_turn")
        graph.add_conditional_edges(
            "prepare_turn",
            self._route_branch,
            {
                "flow_education": "education",
                "flow_report_then_education": "report",
                "flow_report_then_intake": "report",
                "flow_intake_then_medication": "intake",
                "flow_general_intake": "intake",
            },
        )
        graph.add_conditional_edges(
            "report",
            self._after_report,
            {
                "education": "education",
                "intake": "intake",
            },
        )
        graph.add_conditional_edges(
            "intake",
            self._after_intake,
            {
                "medication": "medication",
                "triage": "triage",
            },
        )
        graph.add_conditional_edges(
            "triage",
            self._after_triage,
            {
                "medication": "medication",
            },
        )
        graph.add_conditional_edges(
            "medication",
            self._after_medication,
            {
                "triage_after_medication": "triage_after_medication",
                "rag_summary": "rag_summary",
            },
        )
        graph.add_edge("triage_after_medication", "rag_summary")
        graph.add_edge("education", "rag_summary")
        graph.add_edge("rag_summary", "compose_turn")
        graph.add_edge("compose_turn", "persist_turn")
        graph.add_edge("persist_turn", END)

        self._turn_graph = graph.compile()
        return self._turn_graph

    def run(
        self,
        user_id: str,
        user_text: str,
        age: int | None = None,
        sex: str | None = None,
        image_path: str | None = None,
    ) -> str:
        graph = self._build_graph()
        result = graph.invoke(
            {
                "user_id": user_id,
                "user_text": user_text,
                "age": age,
                "sex": sex,
                "image_path": image_path,
            }
        )
        return result["final_response"]

    def run_visit_turn(
        self,
        user_id: str,
        user_text: str,
        *,
        visit_id: str | None = None,
        age: int | None = None,
        sex: str | None = None,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        existing_visit = self.memory.get_active_visit(user_id, visit_id) if visit_id else None
        graph = self._build_turn_graph()
        started = time.perf_counter()
        result = graph.invoke(
            {
                "user_id": user_id,
                "user_text": user_text,
                "age": age,
                "sex": sex,
                "image_path": image_path,
                "visit_id": visit_id or "",
                "existing_visit": existing_visit,
            }
        )
        turn_result = dict(result["turn_result"])
        turn_result["elapsed_sec"] = time.perf_counter() - started
        return turn_result
