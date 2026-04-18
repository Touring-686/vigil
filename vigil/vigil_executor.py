import logging
from collections.abc import Callable, Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from vigil.config import VIGILConfig
from vigil.runtime_auditor import RuntimeAuditor
from vigil.types import ToolCallInfo

logger = logging.getLogger(__name__)


def vigil_tool_result_to_str(tool_result: FunctionReturnType) -> str:
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

    return tool_result_to_str(tool_result)


class VIGILToolsExecutor(BasePipelineElement):
    def __init__(
        self,
        config: VIGILConfig,
        auditor: RuntimeAuditor,
        tool_output_formatter: Callable[[FunctionReturnType], str] = vigil_tool_result_to_str,
    ):
        self.config = config
        self.auditor = auditor
        self.output_formatter = tool_output_formatter

        # 跟踪每个工具调用的回溯次数
        self._backtracking_counts: dict[str, int] = {}

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        
        # 检查是否有工具调用需要处理
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args

        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        # 处理工具调用
        tool_call_results = []

        for tool_call in messages[-1]["tool_calls"]:
            # 创建工具调用信息
            tool_call_info: ToolCallInfo = {
                "tool_name": tool_call.function,
                "arguments": dict(tool_call.args),
                "tool_call_id": tool_call.id,
            }

            # 检查工具是否存在
            if tool_call.function not in (tool.name for tool in runtime.functions.values()):
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_call.function} provided.",
                    )
                )
                continue

            # 进行安全审计
            audit_result = self.auditor.audit_tool_call(tool_call_info)

            if not audit_result.allowed:
                # 工具调用被拦截
                if self.config.log_audit_decisions:
                    logger.warning(f"[VIGILToolsExecutor] Tool call blocked: {tool_call.function}")

                # 检查回溯次数
                backtrack_key = f"{tool_call.function}:{str(tool_call.args)}"
                backtrack_count = self._backtracking_counts.get(backtrack_key, 0)

                if (
                    self.config.enable_reflective_backtracking
                    and backtrack_count < self.config.max_backtracking_attempts
                ):
                    # 启用回溯：返回反馈消息
                    self._backtracking_counts[backtrack_key] = backtrack_count + 1

                    feedback_message = audit_result.feedback_message or (
                        f"The tool call '{tool_call.function}' was blocked by security constraints. "
                        "Please try a different approach."
                    )

                    # 添加回溯提示
                    if self.config.feedback_verbosity in ["detailed", "verbose"]:
                        feedback_message += (
                            f"\n\nAttempt {backtrack_count + 1}/{self.config.max_backtracking_attempts}. "
                            "Consider alternative tools or different parameters."
                        )

                    tool_call_results.append(
                        ChatToolResultMessage(
                            role="tool",
                            content=[text_content_block_from_string(feedback_message)],
                            tool_call_id=tool_call.id,
                            tool_call=tool_call,
                            error="SecurityConstraintViolation",
                        )
                    )

                    logger.info(
                        f"[VIGILToolsExecutor] Reflective backtracking enabled for {tool_call.function} "
                        f"(attempt {backtrack_count + 1}/{self.config.max_backtracking_attempts})"
                    )
                else:
                    # 超过回溯次数或未启用回溯：返回错误
                    error_message = (
                        audit_result.feedback_message
                        or f"Tool '{tool_call.function}' cannot be executed due to security constraints."
                    )

                    if backtrack_count >= self.config.max_backtracking_attempts:
                        error_message += f"\n\nMaximum backtracking attempts ({self.config.max_backtracking_attempts}) reached."

                    tool_call_results.append(
                        ChatToolResultMessage(
                            role="tool",
                            content=[text_content_block_from_string(error_message)],
                            tool_call_id=tool_call.id,
                            tool_call=tool_call,
                            error="SecurityConstraintViolation",
                        )
                    )

                    logger.warning(
                        f"[VIGILToolsExecutor] Tool call permanently blocked: {tool_call.function}"
                    )

            else:
                # 工具调用被允许，执行它
                if self.config.log_audit_decisions:
                    logger.debug(f"[VIGILToolsExecutor] Tool call allowed: {tool_call.function}")

                # 执行工具
                tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)

                formatted_result = self.output_formatter(tool_call_result)

                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(formatted_result)],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=error,
                    )
                )

                # 成功执行后重置回溯计数
                backtrack_key = f"{tool_call.function}:{str(tool_call.args)}"
                if backtrack_key in self._backtracking_counts:
                    del self._backtracking_counts[backtrack_key]

        return query, runtime, env, [*messages, *tool_call_results], extra_args

    def reset_backtracking_counts(self) -> None:
        self._backtracking_counts.clear()
        logger.debug("[VIGILToolsExecutor] Backtracking counts reset")


class VIGILInitQuery(BasePipelineElement):

    def __init__(self, config: VIGILConfig, constraint_generator, auditor: RuntimeAuditor):
        self.config = config
        self.constraint_generator = constraint_generator
        self.auditor = auditor

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        
        # 检查是否是新的用户查询（只在初始查询时生成约束）
        if len(messages) == 0 or (len(messages) == 1 and messages[0]["role"] == "system"):
            # 生成约束
            logger.info(f"[VIGILInitQuery] Generating constraints for query: {query[:50]}...")
            constraint_set = self.constraint_generator.generate_constraints(query)

            # 更新审计器
            self.auditor.update_constraints(constraint_set)

            logger.info(f"[VIGILInitQuery] Generated {len(constraint_set.constraints)} constraints")

            # 可选：在extra_args中保存约束集供后续使用
            extra_args = {**extra_args, "vigil_constraint_set": constraint_set}

        # 添加用户消息
        from agentdojo.types import ChatUserMessage, text_content_block_from_string

        query_message = ChatUserMessage(role="user", content=[text_content_block_from_string(query)])
        messages = [*messages, query_message]

        return query, runtime, env, messages, extra_args
