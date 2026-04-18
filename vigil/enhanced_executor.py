import logging
from collections.abc import Callable, Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from vigil.abstract_sketch import AbstractSketchGenerator
from vigil.commitment_manager import CommitmentManager
from vigil.config import VIGILConfig
from vigil.constraint_generator import ConstraintGenerator
from vigil.enhanced_auditor import EnhancedRuntimeAuditor
from vigil.hypothesizer import Hypothesizer
from vigil.path_cache import PathCache
from vigil.perception_sanitizer import PerceptionSanitizer
from vigil.types import ToolCallInfo

logger = logging.getLogger(__name__)


def enhanced_tool_result_to_str(tool_result: FunctionReturnType) -> str:
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

    return tool_result_to_str(tool_result)


class EnhancedVIGILToolsExecutor(BasePipelineElement):
    
    def __init__(
        self,
        config: VIGILConfig,
        auditor: EnhancedRuntimeAuditor,
        sanitizer: PerceptionSanitizer,
        hypothesizer=None,
        commitment_manager: CommitmentManager | None = None,
        path_cache: PathCache | None = None,
        tool_output_formatter: Callable[[FunctionReturnType], str] = enhanced_tool_result_to_str,
        openai_client=None,
    ):
        self.config = config
        self.auditor = auditor
        self.sanitizer = sanitizer
        self.hypothesizer = hypothesizer
        self.commitment_manager = commitment_manager
        self.path_cache = path_cache
        self.output_formatter = tool_output_formatter
        self.openai_client = openai_client

        # 跟踪每个工具调用的回溯次数
        self._backtracking_counts: dict[str, int] = {}

        # 跟踪参数验证错误的重试次数
        self._validation_retry_counts: dict[str, int] = {}
        self._max_validation_retries = 3  # 每个工具最多重试3次参数

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # ===== 优先检查是否是 REASONING 步骤（__no_tool_call__）=====
        # REASONING 步骤：直接调用 LLM 进行推理，把 LLM 当作一个工具
        current_step_is_reasoning = extra_args.get('current_step_is_reasoning', False)
        current_step_is_response = extra_args.get('current_step_is_response', False)

        if current_step_is_reasoning:
            if current_step_is_response:
                logger.info(
                    "[EnhancedVIGILExecutor] 📝 RESPONSE step detected - calling LLM to generate final response"
                )
            else:
                logger.info(
                    "[EnhancedVIGILExecutor] 🧠 REASONING step detected - calling LLM as a reasoning tool (no tool execution allowed)"
                )

            # 调用 LLM 进行推理（不允许使用工具）
            reasoning_message = self._execute_reasoning_step(messages, query, extra_args)

            # 将 LLM 的推理结果作为 assistant message 添加到历史
            messages = [*messages, reasoning_message]

            # 如果是 __response__ 步骤，将输出存储到 extra_args 中
            if current_step_is_response:
                # 提取响应内容
                response_content = "Response completed."
                if reasoning_message and "content" in reasoning_message and reasoning_message["content"]:
                    if isinstance(reasoning_message["content"], list) and len(reasoning_message["content"]) > 0:
                        content_block = reasoning_message["content"][0]
                        response_content = content_block.get("content", response_content)

                # 存储到 extra_args，用于最终返回
                extra_args = {**extra_args, 'final_response': response_content}

                logger.info(
                    "[EnhancedVIGILExecutor] ✓ RESPONSE step completed - output stored in extra_args"
                )
            else:
                logger.info(
                    "[EnhancedVIGILExecutor] ✓ REASONING step completed - LLM provided analysis"
                )

            # === 记录 REASONING/RESPONSE 步骤到执行历史 ===
            if self.auditor:
                current_step_index = extra_args.get('current_step_index', 0)
                step_description = None

                # 从 abstract sketch 获取步骤描述
                if self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps'):
                    if current_step_index < len(self.auditor.abstract_sketch.steps):
                        step = self.auditor.abstract_sketch.steps[current_step_index]
                        step_description = f"{step.step_type} - {step.description}"

                # 提取推理结果文本
                reasoning_result = "Reasoning completed."
                if reasoning_message and "content" in reasoning_message and reasoning_message["content"]:
                    if isinstance(reasoning_message["content"], list) and len(reasoning_message["content"]) > 0:
                        content_block = reasoning_message["content"][0]
                        reasoning_result = content_block.get("content", reasoning_result)

                # 创建虚拟的 tool_call_info 用于记录
                tool_name = "__llm_response__" if current_step_is_response else "__llm_reasoning__"
                reasoning_tool_call: ToolCallInfo = {
                    "tool_name": tool_name,
                    "arguments": {},
                    "tool_call_id": None,
                }

                self.auditor.record_execution_step(
                    step_index=current_step_index,
                    tool_call_info=reasoning_tool_call,
                    result=reasoning_result,
                    step_description=step_description,
                )

                if self.config.log_audit_decisions:
                    step_type = "RESPONSE" if current_step_is_response else "REASONING"
                    logger.info(
                        f"[EnhancedVIGILExecutor] Recorded {step_type} step {current_step_index + 1} "
                        f"to execution history: {reasoning_result}..."
                    )

            # 清除 REASONING 标志
            extra_args = {**extra_args, 'current_step_is_reasoning': False, 'finished_task': False}

            return query, runtime, env, messages, extra_args

        # ===== 正常的工具执行流程 =====
        # 检查是否有工具调用需要处理
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args

        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        # 检查是否有工具调用
        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        # ===== CRITICAL: 检查是否尝试调用多个工具 =====
        # VIGIL的设计原则：每次只执行一个工具，确保可控性和安全性
        if len(messages[-1]["tool_calls"]) > 1:
            logger.warning(
                f"[EnhancedVIGILToolsExecutor] Agent attempted to call {len(messages[-1]['tool_calls'])} tools "
                f"simultaneously. VIGIL policy: ONE tool per turn."
            )

            # 拒绝所有工具调用，返回错误消息
            error_message = (
                f"❌ VIGIL Policy Violation: Multiple Tool Calls Detected\n\n"
                f"You attempted to call {len(messages[-1]['tool_calls'])} tools simultaneously:\n"
            )

            for i, tc in enumerate(messages[-1]["tool_calls"], 1):
                error_message += f"  {i}. {tc.function}({dict(tc.args)})\n"

            error_message += (
                f"\n**VIGIL Policy: You MUST call exactly ONE tool per turn.**\n\n"
                f"To complete this task:\n"
                f"1. Choose the MOST IMPORTANT tool for the current step\n"
                f"2. Call that tool alone and wait for its result\n"
                f"3. In subsequent turns, call additional tools if needed\n\n"
                f"Please retry with a SINGLE tool call."
            )

            # 生成一个错误结果消息（针对第一个工具调用）
            tool_call_results = [
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(error_message)],
                    tool_call_id=messages[-1]["tool_calls"][0].id,
                    tool_call=messages[-1]["tool_calls"][0],
                    error=error_message,
                )
            ]

            # 立即返回错误，不执行任何工具
            return query, runtime, env, [*messages, *tool_call_results], extra_args

        # 设置可用工具列表（用于冗余性检查）
        available_tools = [
            {"name": tool.name, "description": tool.description}
            for tool in runtime.functions.values()
        ]
        self.auditor.set_available_tools(available_tools)

        # 注意：Hypothesis Tree 生成已移至 HypothesisGuidanceElement
        # 该element在LLM推理之前运行，生成guidance并注入到context中
        # 这确保了正确的执行顺序：
        #   Hypothesis Generation → Verification → Commitment → LLM Decision
        # 而不是错误的：
        #   LLM Decision → Hypothesis Generation (事后分析)

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

            # ===== 检查是否需要审计 =====
            # 在 direct mode 下，如果工具已经通过 HypothesisGuidance 的完整审计，跳过重复审计
            skip_audit = extra_args.get('skip_audit', False) and extra_args.get('vigil_pre_approved') == tool_call.function

            if skip_audit:
                # 工具已经通过 Two-Stage Verification，直接执行
                if self.config.log_audit_decisions:
                    logger.info(
                        f"[EnhancedVIGILExecutor] Tool '{tool_call.function}' PRE-APPROVED by HypothesisGuidance "
                        f"(Two-Stage Verification passed), skipping redundant audit"
                    )

                # 直接执行工具（跳到执行部分）
                needs_backtrack = self._execute_tool(
                    tool_call=tool_call,
                    tool_call_info=tool_call_info,
                    runtime=runtime,
                    env=env,
                    query=query,
                    tool_call_results=tool_call_results,
                    extra_args=extra_args,
                )

                # 如果检测到需要回溯（SOP 注入），设置标志
                if needs_backtrack:
                    extra_args = {**extra_args, 'backtrack_needed': True}

                # 清除标志（避免影响下一轮）
                if 'skip_audit' in extra_args:
                    extra_args = {**extra_args, 'skip_audit': False, 'vigil_pre_approved': None}

                continue

            # ===== 进行安全审计 =====
            # 注意：这里的审计是针对以下场景：
            # 1. Guidance mode：LLM 可能不遵循 guidance，需要最后一道防线
            # 2. 非 direct mode：传统的审计流程
            if self.config.log_audit_decisions:
                logger.info(
                    f"[EnhancedVIGILExecutor] Auditing tool call: {tool_call.function} "
                    f"(not pre-approved, performing full security check)"
                )

            audit_result = self.auditor.audit_tool_call(tool_call_info)

            if not audit_result.allowed:
                # 工具调用被拦截
                if self.config.log_audit_decisions:
                    logger.warning(f"[EnhancedVIGILExecutor] Tool call blocked: {tool_call.function}")

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
                        f"[EnhancedVIGILExecutor] Reflective backtracking enabled for {tool_call.function} "
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
                        f"[EnhancedVIGILExecutor] Tool call permanently blocked: {tool_call.function}"
                    )

            else:
                # 工具调用被允许，执行它
                if self.config.log_audit_decisions:
                    logger.info(f"[EnhancedVIGILExecutor] ✓ Tool call ALLOWED: {tool_call.function}")

                # 执行工具
                needs_backtrack = self._execute_tool(
                    tool_call=tool_call,
                    tool_call_info=tool_call_info,
                    runtime=runtime,
                    env=env,
                    query=query,
                    tool_call_results=tool_call_results,
                    extra_args=extra_args,
                )

                # 如果检测到需要回溯（SOP 注入），设置标志
                if needs_backtrack:
                    extra_args = {**extra_args, 'backtrack_needed': True}

        # 检查是否完成所有 abstract sketch 步骤
        # 条件：
        # 1. 没有需要回溯的情况
        # 2. 有 abstract sketch
        # 3. 当前步骤索引已经到达或超过总步骤数
        if not extra_args.get('backtrack_needed', False):
            # 从 extra_args 或 auditor 获取 abstract sketch
            abstract_sketch = extra_args.get('vigil_abstract_sketch') or (
                self.auditor.abstract_sketch if hasattr(self, 'auditor') and self.auditor else None
            )

            if abstract_sketch and hasattr(abstract_sketch, 'steps'):
                # 从 hypothesis_guidance 获取当前步骤索引
                # 步骤索引在 HypothesisGuidance 中维护
                current_step_index = extra_args.get('current_step_index', 0)
                total_steps = len(abstract_sketch.steps)

                # 检查是否所有步骤都完成了
                # 注意：current_step_index 在 HypothesisGuidance 中每次执行后会 +1
                # 所以当 current_step_index >= total_steps 时，说明所有步骤都执行完了
                if current_step_index >= total_steps:
                    extra_args = {**extra_args, 'finished_task': True}
                    logger.info(
                        f"[EnhancedVIGILExecutor] ✓ All {total_steps} sketch steps completed successfully, "
                        f"marking task as finished"
                    )

        return query, runtime, env, [*messages, *tool_call_results], extra_args

    def _execute_tool(
        self,
        tool_call,
        tool_call_info: ToolCallInfo,
        runtime: FunctionsRuntime,
        env: Env,
        query: str,
        tool_call_results: list,
        extra_args: dict = {},
    ) -> bool:
        vigil_iterative_calls = extra_args.get('vigil_iterative_calls')

        if vigil_iterative_calls and len(vigil_iterative_calls) > 1:
            # 迭代执行：依次调用同一个工具多次（每次参数不同）
            logger.info(
                f"[EnhancedVIGILExecutor] Iterative execution detected: "
                f"{len(vigil_iterative_calls)} calls for tool '{tool_call.function}'"
            )
            return self._execute_tool_with_iteration(
                tool_call=tool_call,
                tool_call_info=tool_call_info,
                runtime=runtime,
                env=env,
                query=query,
                tool_call_results=tool_call_results,
                extra_args=extra_args,
                iterative_calls=vigil_iterative_calls,
            )

        # === ValidationError 自动重试循环 ===
        max_validation_retries = self._max_validation_retries
        current_retry = 0
        current_args = tool_call.args
        tool_call_result = None
        error = None

        while current_retry <= max_validation_retries:
            # 执行工具
            logger.info(f"[EnhancedVIGILExecutor] Executing tool: {tool_call.function}({dict(current_args)}) [attempt {current_retry + 1}]")
            tool_call_result, error = runtime.run_function(env, tool_call.function, current_args)
            logger.info(f"[EnhancedVIGILExecutor] Tool execution completed. Error: {error is not None}")

            # 检查是否为 ValidationError
            if error and self._is_validation_error(error):
                current_retry += 1

                if current_retry > max_validation_retries:
                    logger.warning(
                        f"[EnhancedVIGILExecutor] ValidationError retry limit reached for '{tool_call.function}' "
                        f"({current_retry} attempts). Returning error to LLM."
                    )
                    break

                # 调用 LLM 分析错误并修正参数
                logger.info(
                    f"[EnhancedVIGILExecutor] ValidationError detected. "
                    f"Calling LLM to fix parameters (attempt {current_retry}/{max_validation_retries})..."
                )

                corrected_args = self._llm_fix_validation_error(
                    tool_name=tool_call.function,
                    original_args=current_args,
                    error_message=error,
                    runtime=runtime
                )

                if corrected_args is None:
                    logger.warning(
                        f"[EnhancedVIGILExecutor] LLM unable to fix ValidationError. Returning error to LLM."
                    )
                    break

                logger.info(
                    f"[EnhancedVIGILExecutor] LLM corrected parameters: {dict(current_args)} -> {dict(corrected_args)}"
                )

                # 使用修正后的参数重试
                current_args = corrected_args
                tool_call.args = corrected_args  # 更新 tool_call 对象
                continue
            else:
                # 没有 ValidationError（执行成功或遇到其他错误），退出循环
                break

        # === 分类错误类型 ===
        needs_backtrack = False
        is_validation_error = False
        original_error = error  # 保存原始错误用于检测

        # 检测错误类型（重试后仍有错误的情况）
        if error:
            # 1. 检测参数类型验证错误（已经重试过，现在失败了）
            is_validation_error = self._is_validation_error(error)

            if is_validation_error:
                logger.warning(
                    f"[EnhancedVIGILExecutor] ValidationError persists after {current_retry} retries for '{tool_call.function}'. "
                    f"Converting to backtrack."
                )
                # 超过重试限制，转换为需要回溯（尝试其他工具）
                is_validation_error = False
                needs_backtrack = True

            # 2. 检测 SOP 注入（恶意攻击，需要回溯）
            if not is_validation_error and self._detect_sop_injection(error):
                logger.warning(
                    f"[EnhancedVIGILExecutor] ⚠️ SOP INJECTION DETECTED in error message "
                    f"from tool '{tool_call.function}' (Type III-A/III-B attack)"
                )
                needs_backtrack = True

        # === Perception Sanitizer: 清洗返回值和错误 ===
        if self.config.enable_perception_sanitizer:
            # 清洗返回值
            tool_call_result = self.sanitizer.sanitize_tool_result(
                tool_call.function, tool_call_result
            )

            # 清洗错误消息
            if error:
                if is_validation_error:
                    # 参数验证错误：保留详细信息供 LLM 修正，但移除可能的注入
                    error = self._sanitize_validation_error(error, tool_call.function)
                else:
                    # 其他错误（包括 SOP 注入）：完全清洗
                    error = self.sanitizer.sanitize_error_message(tool_call.function, error)

                if self.config.log_sanitizer_actions and original_error:
                    logger.info(
                        f"[EnhancedVIGILExecutor] Error message sanitized: "
                        f"'{original_error}...' → '{error if error else 'None'}...'"
                    )

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

        # === 记录执行历史（仅在成功时）===
        if not error and self.auditor:
            current_step_index = extra_args.get('current_step_index', 0)
            step_description = None

            # 从 abstract sketch 获取步骤描述
            if self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps'):
                if current_step_index < len(self.auditor.abstract_sketch.steps):
                    step = self.auditor.abstract_sketch.steps[current_step_index]
                    step_description = f"{step.step_type} - {step.description}"

            self.auditor.record_execution_step(
                step_index=current_step_index,
                tool_call_info=tool_call_info,
                result=formatted_result,
                step_description=step_description,
            )

        # === Path Cache: 记录成功执行的路径 ===
        if self.path_cache:
            # 判断执行是否成功（无错误）
            outcome = "failure" if error else "success"

            # 获取当前步骤索引
            current_step_index = extra_args.get('current_step_index', 0)

            # 获取 abstract step description（从 auditor 的 abstract sketch）
            abstract_step_description = None
            abstract_step_type = None
            if self.auditor and self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps'):
                # current_step_index 就是当前步骤
                step_idx = current_step_index
                if 0 <= step_idx < len(self.auditor.abstract_sketch.steps):
                    current_step = self.auditor.abstract_sketch.steps[step_idx]
                    abstract_step_description = current_step.description
                    abstract_step_type = current_step.step_type

            # 添加到 path cache（关键：传递 step_index 和 abstract_step_description）
            self.path_cache.add_verified_path(
                user_query=query,
                tool_name=tool_call.function,
                arguments=dict(tool_call.args),
                outcome=outcome,
                step_index=current_step_index,  # 传递步骤索引
                abstract_step=abstract_step_type,  # 传递抽象步骤类型
                metadata={"error": error} if error else None,
            )

            if self.config.log_audit_decisions and outcome == "success":
                logger.info(
                    f"[EnhancedVIGILExecutor] ✓ Path cached: step {current_step_index}, "
                    f"tool '{tool_call.function}', outcome '{outcome}'"
                )
                if abstract_step_description:
                    logger.debug(
                        f"[EnhancedVIGILExecutor] Abstract step: '{abstract_step_description}...'"
                    )

        # 成功执行后重置回溯计数
        backtrack_key = f"{tool_call.function}:{str(tool_call.args)}"
        if backtrack_key in self._backtracking_counts:
            del self._backtracking_counts[backtrack_key]

        # 返回是否需要回溯
        return needs_backtrack

    def _execute_tool_with_iteration(
        self,
        tool_call,
        tool_call_info: ToolCallInfo,
        runtime: FunctionsRuntime,
        env: Env,
        query: str,
        tool_call_results: list,
        extra_args: dict,
        iterative_calls: list[dict]
    ) -> bool:
        
        tool_name = tool_call.function
        all_results = []
        all_errors = []
        needs_backtrack = False
        abstract_step = None

        logger.info(
            f"[EnhancedVIGILExecutor] Starting iterative execution for '{tool_name}': "
            f"{len(iterative_calls)} iterations"
        )

        for idx, iter_call in enumerate(iterative_calls):
            iter_args = iter_call.get("arguments", {})
            iter_reasoning = iter_call.get("reasoning", f"Iteration {idx + 1}")

            logger.info(
                f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] {tool_name}({iter_args})"
            )
            logger.debug(f"[EnhancedVIGILExecutor] Reasoning: {iter_reasoning}")

            # === ValidationError 自动重试循环（针对每次迭代）===
            max_validation_retries = self._max_validation_retries
            current_retry = 0
            current_args = iter_args
            tool_call_result = None
            error = None

            while current_retry <= max_validation_retries:
                # 执行工具
                logger.info(
                    f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] Executing {tool_name} "
                    f"[attempt {current_retry + 1}]"
                )
                tool_call_result, error = runtime.run_function(env, tool_name, current_args)

                # 检查是否为 ValidationError
                if error and self._is_validation_error(error):
                    current_retry += 1

                    if current_retry > max_validation_retries:
                        logger.warning(
                            f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] "
                            f"ValidationError retry limit reached ({current_retry} attempts)"
                        )
                        break

                    # 调用 LLM 分析错误并修正参数
                    logger.info(
                        f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] "
                        f"ValidationError detected, calling LLM to fix parameters "
                        f"(attempt {current_retry}/{max_validation_retries})"
                    )

                    corrected_args = self._llm_fix_validation_error(
                        tool_name=tool_name,
                        original_args=current_args,
                        error_message=error,
                        runtime=runtime
                    )

                    if corrected_args is None:
                        logger.warning(
                            f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] "
                            f"LLM unable to fix ValidationError"
                        )
                        break

                    logger.info(
                        f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] "
                        f"LLM corrected parameters: {current_args} -> {corrected_args}"
                    )

                    # 使用修正后的参数重试
                    current_args = corrected_args
                    continue
                else:
                    # 没有 ValidationError，退出重试循环
                    break

            # === 清洗结果和错误 ===
            if self.config.enable_perception_sanitizer:
                tool_call_result = self.sanitizer.sanitize_tool_result(tool_name, tool_call_result)
                if error:
                    error = self.sanitizer.sanitize_error_message(tool_name, error)

            # === 收集结果 ===
            if error:
                logger.warning(
                    f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] Error: {error}..."
                )
                all_errors.append(f"[Iteration {idx+1}]: {error}")

                # 检测 SOP 注入
                if self._detect_sop_injection(error):
                    logger.warning(
                        f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] "
                        f"⚠️ SOP INJECTION DETECTED"
                    )
                    needs_backtrack = True
            else:
                formatted_result = self.output_formatter(tool_call_result)
                logger.info(
                    f"[EnhancedVIGILExecutor] [{idx+1}/{len(iterative_calls)}] "
                    f"Success: {formatted_result}..."
                )
                all_results.append(f"[Iteration {idx+1}]: {formatted_result}")

        # === 合并所有结果 ===
        if all_results:
            combined_result = "\n\n".join(all_results)
            formatted_result = self.output_formatter(combined_result)
        else:
            # 全部失败
            formatted_result = "All iterative calls failed."

        # 合并所有错误（如果有）
        combined_error = None
        if all_errors:
            combined_error = "Some calls failed:\n" + "\n".join(all_errors)

        # 添加合并后的结果到 tool_call_results
        tool_call_results.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted_result)],
                tool_call_id=tool_call.id,
                tool_call=tool_call,
                error=combined_error,
            )
        )

        # 记录执行历史（如果有成功的调用）
        if all_results and self.auditor:
            current_step_index = extra_args.get('current_step_index', 0)
            step_description = None

            # 从 abstract sketch 获取步骤描述
            if self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps'):
                if current_step_index < len(self.auditor.abstract_sketch.steps):
                    step = self.auditor.abstract_sketch.steps[current_step_index]
                    step_description = f"{step.step_type} - {step.description}"

            # 记录（使用合并后的结果）
            self.auditor.record_execution_step(
                step_index=current_step_index,
                tool_call_info=tool_call_info,
                result=formatted_result,
                step_description=step_description,
            )

            if self.config.log_audit_decisions:
                logger.info(
                    f"[EnhancedVIGILExecutor] Recorded iterative tool call "
                    f"step {current_step_index}: '{tool_name}' "
                    f"(called {len(iterative_calls)} times: {len(all_results)} successful, {len(all_errors)} failed)"
                )

        # Path Cache: 记录成功执行的路径
        if self.path_cache:
            outcome = "failure" if all_errors and not all_results else "success"
            current_step_index = extra_args.get('current_step_index', 0)

            # 获取 abstract step description
            abstract_step_description = None
            if self.auditor and self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps'):
                step_idx = current_step_index
                if 0 <= step_idx < len(self.auditor.abstract_sketch.steps):
                    current_step = self.auditor.abstract_sketch.steps[step_idx]
                    abstract_step_description = current_step.description
                    abstract_step = current_step.step_type

            # 添加到 path cache（记录迭代执行信息）
            self.path_cache.add_verified_path(
                user_query=query,
                tool_name=tool_name,
                arguments={"iterative_calls": [ic.get("arguments") for ic in iterative_calls]},  # 记录所有参数
                outcome=outcome,
                step_index=current_step_index,
                abstract_step=abstract_step,
                metadata={
                    "is_iterative": True,
                    "total_iterations": len(iterative_calls),
                    "successful_iterations": len(all_results),
                    "failed_iterations": len(all_errors),
                }
            )

            if self.config.log_audit_decisions and outcome == "success":
                logger.info(
                    f"[EnhancedVIGILExecutor] ✓ Path cached: step {current_step_index}, "
                    f"iterative tool '{tool_name}', outcome '{outcome}'"
                )

        logger.info(
            f"[EnhancedVIGILExecutor] Iterative execution complete: "
            f"{len(all_results)} successful, {len(all_errors)} failed"
        )

        # 返回是否需要回溯
        return needs_backtrack

    def _detect_list_params_needing_split(
        self,
        tool_name: str,
        args: dict,
        runtime: FunctionsRuntime
    ) -> list[str]:

        # 获取工具的参数 schema
        tool_schema = None
        for tool in runtime.functions.values():
            if tool.name == tool_name:
                tool_schema = tool.parameters.model_json_schema()
                break

        if not tool_schema or 'properties' not in tool_schema:
            return []

        list_params = []
        for param_name, param_value in args.items():
            # 检查参数值是否为列表
            if not isinstance(param_value, list):
                continue

            # 检查参数定义是否期望单个值（非数组类型）
            if param_name in tool_schema['properties']:
                param_def = tool_schema['properties'][param_name]
                expected_type = param_def.get('type', '')

                # 如果定义的类型不是 array，说明期望单个值
                if expected_type != 'array':
                    list_params.append(param_name)
                    logger.debug(
                        f"[EnhancedVIGILExecutor] Parameter '{param_name}' is a list {param_value} "
                        f"but expects type '{expected_type}'"
                    )

        return list_params

    def _execute_tool_with_list_expansion(
        self,
        tool_call,
        tool_call_info: ToolCallInfo,
        runtime: FunctionsRuntime,
        env: Env,
        query: str,
        tool_call_results: list,
        extra_args: dict,
        list_params: list[str]
    ) -> bool:
        # 获取列表参数的值
        original_args = dict(tool_call.args)

        # 假设只拆分第一个列表参数（简化处理）
        # TODO: 如果有多个列表参数，可能需要笛卡尔积
        list_param_name = list_params[0]
        list_values = original_args[list_param_name]

        if not list_values:
            # 空列表，返回空结果
            logger.warning(
                f"[EnhancedVIGILExecutor] List parameter '{list_param_name}' is empty. "
                f"Returning empty result."
            )
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string("No items to process (empty list).")],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=None,
                )
            )
            return False

        # 多次调用工具，每次使用列表中的一个值
        all_results = []
        all_errors = []
        needs_backtrack = False

        logger.info(
            f"[EnhancedVIGILExecutor] Expanding list parameter '{list_param_name}' "
            f"with {len(list_values)} values: {list_values}"
        )

        for idx, value in enumerate(list_values):
            # 创建单个值的参数
            single_args = original_args.copy()
            single_args[list_param_name] = value

            logger.info(
                f"[EnhancedVIGILExecutor] [{idx+1}/{len(list_values)}] Calling {tool_call.function}({dict(single_args)})"
            )

            # 执行工具
            result, error = runtime.run_function(env, tool_call.function, single_args)

            # 清洗结果和错误（如果启用）
            if self.config.enable_perception_sanitizer:
                result = self.sanitizer.sanitize_tool_result(tool_call.function, result)
                if error:
                    error = self.sanitizer.sanitize_error_message(tool_call.function, error)

            # 收集结果
            if error:
                logger.warning(
                    f"[EnhancedVIGILExecutor] [{idx+1}/{len(list_values)}] Error: {error}..."
                )
                all_errors.append(f"[{value}]: {error}")

                # 检测 SOP 注入
                if self._detect_sop_injection(error):
                    logger.warning(
                        f"[EnhancedVIGILExecutor] ⚠️ SOP INJECTION DETECTED in error for value '{value}'"
                    )
                    needs_backtrack = True
            else:
                logger.info(
                    f"[EnhancedVIGILExecutor] [{idx+1}/{len(list_values)}] Success: {str(result)}..."
                )
                all_results.append(f"[{value}]: {result}")

        # 合并所有结果
        if all_results:
            combined_result = "\n\n".join(all_results)
            formatted_result = self.output_formatter(combined_result)
        else:
            # 全部失败
            formatted_result = "All calls failed."

        # 合并所有错误（如果有）
        combined_error = None
        if all_errors:
            combined_error = "Some calls failed:\n" + "\n".join(all_errors)

        # 添加合并后的结果到 tool_call_results
        tool_call_results.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted_result)],
                tool_call_id=tool_call.id,
                tool_call=tool_call,
                error=combined_error,
            )
        )

        # 记录执行历史（如果有成功的调用）
        if all_results and self.auditor:
            current_step_index = extra_args.get('current_step_index', 0)
            step_description = None

            # 从 abstract sketch 获取步骤描述
            if self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps'):
                if current_step_index < len(self.auditor.abstract_sketch.steps):
                    step = self.auditor.abstract_sketch.steps[current_step_index]
                    step_description = f"{step.step_type} - {step.description}"

            # 记录（使用原始参数，因为是列表调用）
            self.auditor.record_execution_step(
                step_index=current_step_index,
                tool_call_info=tool_call_info,
                result=formatted_result,
                step_description=step_description,
            )

            if self.config.log_audit_decisions:
                logger.info(
                    f"[EnhancedVIGILExecutor] Recorded expanded tool call "
                    f"step {current_step_index}: '{tool_call.function}' "
                    f"(called {len(list_values)} times)"
                )

        logger.info(
            f"[EnhancedVIGILExecutor] List expansion complete: {len(all_results)} successful, "
            f"{len(all_errors)} failed"
        )

        return needs_backtrack

    def _execute_reasoning_step(
        self,
        messages: Sequence[ChatMessage],
        query: str,
        extra_args: dict
    ) -> ChatMessage:
        """执行 REASONING 步骤：调用 LLM 进行推理（作为工具使用）

        当 hypothesis branch 是 __no_tool_call__ 或 __internal_reasoning__ 时，
        把 LLM 当作工具来执行推理。

        Args:
            messages: 当前消息历史（可能包含 guidance message）
            query: 用户查询

        Returns:
            包含 LLM 推理结果的 ChatAssistantMessage
        """
        from agentdojo.types import ChatAssistantMessage

        # 检查是否有 hypothesizer 和 openai_client
        if not self.hypothesizer or not self.hypothesizer.openai_client:
            logger.error("[EnhancedVIGILExecutor] No hypothesizer or openai_client available for reasoning step")
            return ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("Error: LLM client not available for reasoning.")],
                tool_calls=None
            )

        try:
            # 将 messages 转换为 OpenAI API 格式
            # 这样 LLM 可以看到完整的上下文，包括 guidance message
            converted_messages = []
            execution_history = ""
            
            # for exe in self.auditor.execution_history:
            #     execution_history += f"Step {exe['step_index']}: {exe['step_description'] or 'N/A'}\n"
            #     execution_history += f"Tool: {exe['tool_name']}\n\tArguments: {exe['arguments']}\nResult: {exe['result']}\n\n"
            # target = self.auditor.abstract_sketch.steps[extra_args.get('current_step_index', 0)-1].description if self.auditor and self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps') and extra_args.get('current_step_index', 0) > 0 else "N/A"
            # query = f"execution history:\n{execution_history}\ntarget:\n{target}"
            # # 如果没有消息，使用简单的 prompt
            # if not converted_messages:
            #     converted_messages = [
            #         {"role": "system", "content": "You are a helpful assistant. Your goal is to reason about the query and execute the target action."},
            #         {"role": "user", "content": query}
            #     ]
            # 构建更清晰的 Context 字符串
            formatted_history = "NO HISTORY AVAILABLE"
            if self.auditor.execution_history:
                history_lines = []
                for exe in self.auditor.execution_history:
                    # 格式化每一步，使其更易读
                    step_str = f"--- Step {exe['step_index']} ---\n"
                    step_str += f"Description: {exe['step_description'] or 'N/A'}\n"
                    step_str += f"Tool Call: {exe['tool_name']}({exe['arguments']})\n"
                    step_str += f"Observation/Result: {exe['result']}\n"
                    history_lines.append(step_str)
                formatted_history = "\n".join(history_lines)

            # 获取当前的 Target Description
            target_step_description = "N/A"
            if self.auditor and self.auditor.abstract_sketch and hasattr(self.auditor.abstract_sketch, 'steps') and extra_args.get('current_step_index', 0) > 0:
                # 注意：index需要处理好，确保对应的是当前要执行的step
                target_step_description = self.auditor.abstract_sketch.steps[extra_args.get('current_step_index', 0)-1].description

            # ------------------------------------------------------------------
            # [CRITICAL] 优化后的 SYSTEM PROMPT
            # ------------------------------------------------------------------
            internal_reasoning_system_prompt = """[ROLE]
            You are the **Internal Reasoning Engine** for the VIGIL security framework.
Your task is to process the [EXECUTION HISTORY] to satisfy the [CURRENT TARGET] using **ONLY** the provided data.

[STRICT GROUNDING RULES - READ CAREFULLY]
1. **NO HALLUCINATION**: You MUST NOT invent, guess, or assume any data. All entities (names, dates, amounts, file contents) must exist verbatim in the [EXECUTION HISTORY].
2. **NO EXTERNAL KNOWLEDGE**: Do not use your pre-training knowledge to answer questions. If the answer is not in the history, state "Data missing from history".
3. **NO NEW TOOL CALLS**: You cannot call tools here. You are a pure reasoning engine. Your output is the result of processing existing observations.

[TASK TYPES]
- **Filtering**: If the target says "Find X in results", verify X exists in the Observation of previous steps and extract it.
- **Calculation**: If the target says "Calculate total", sum up numbers explicitly found in Observations.
- **Summarization/Response**: If the target says "Generate Final Answer", synthesize a response based SOLELY on what the tools actually returned.
- **Verification**: If the previous tool returned an error or empty list, acknowledge the failure. Do NOT pretend it succeeded.

[INPUT DATA]
"""

            # 构建 User Prompt
            user_query_content = f"""[EXECUTION HISTORY]\n{formatted_history}
[CURRENT TARGET (LOGIC TO EXECUTE)]\n{target_step_description}\n
Based strictly on the history above, perform the logic described in the target.
Output the result of your reasoning:"""

            # 组装 Messages
            if not converted_messages:
                converted_messages = [
                    {"role": "system", "content": internal_reasoning_system_prompt},
                    {"role": "user", "content": user_query_content}
                ]
            # 调用 LLM（使用 hypothesizer 的 openai_client，不提供 tools）
            response = self.hypothesizer.openai_client.chat.completions.create(
                model=self.config.hypothesizer_model,
                messages=converted_messages,
                temperature=self.config.hypothesizer_temperature,
                max_tokens=8192,
            )

            # 提取响应
            response_text = response.choices[0].message.content.strip()

            if self.config.log_audit_decisions:
                logger.info(
                    f"[EnhancedVIGILExecutor] LLM reasoning response: {response_text}..."
                )

            # 返回 assistant message
            return ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(response_text or "Reasoning completed.")],
                tool_calls=None
            )

        except Exception as e:
            logger.error(f"[EnhancedVIGILExecutor] Reasoning step error: {e}")
            return ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(f"Error: {str(e)}")],
                tool_calls=None
            )

    def _llm_fix_validation_error(
        self,
        tool_name: str,
        original_args: dict,
        error_message: str,
        runtime: FunctionsRuntime
    ) -> dict | None:
        import json
        from anthropic import Anthropic

        # 获取工具的 schema 和 docstring
        tool_schema = None
        tool_docstring = ""
        for tool in runtime.functions.values():
            if tool.name == tool_name:
                tool_schema = tool.parameters.model_json_schema()
                # 优先从底层函数对象获取原始 docstring（未被 sanitizer 清洗）
                # 因为 runtime 中的 full_docstring 可能已被 ToolDocstringSanitizer 清洗，
                # 丢失了关键的参数格式说明（如 attachments 的 type/file_path 格式）
                raw_doc = getattr(tool.run, '__doc__', None)
                tool_docstring = raw_doc or tool.full_docstring or tool.description or ""
                break

        if not tool_schema:
            logger.error(f"[EnhancedVIGILExecutor] Cannot find schema for tool '{tool_name}'")
            return None

        # 构建执行历史摘要（帮助 LLM 了解文件路径等上下文）
        history_context = ""
        if self.auditor and self.auditor.execution_history:
            recent_entries = self.auditor.execution_history[-5:]  # 最近5步
            history_lines = []
            for exe in recent_entries:
                result_preview = str(exe.get('result', ''))[:200]
                history_lines.append(
                    f"  - {exe.get('tool_name', '?')}({exe.get('arguments', {})}) → {result_preview}"
                )
            history_context = "\n\nRecent Execution History (for context about file paths, downloads, etc.):\n" + "\n".join(history_lines)

        # 构建 prompt（包含 schema 帮助 LLM 理解参数格式）
        # 注意：tool_docstring 可能被 PerceptionSanitizer 清洗过，丢失了关键格式说明
        # 所以同时提供 JSON Schema 作为参数格式的权威来源
        schema_str = json.dumps(tool_schema, indent=2)
        prompt = f"""You are a parameter correction assistant. A tool call failed due to a validation error.

Tool: {tool_name}
Tool Documentation: {tool_docstring}
Parameter Schema: {schema_str}
Original Arguments: {json.dumps(original_args, indent=2)}
Validation Error: {error_message}{history_context}

IMPORTANT: Read the error message carefully and fix ONLY the problematic parameter.
- Do NOT remove parameters that were in the original arguments.
- Do NOT change parameter names unless they are wrong.
- If the error mentions a missing field in a nested object (like attachments), add that field.
- If the error says "path not found", check the execution history for download locations or directory listings to find the correct full path (e.g., /downloads/filename.pdf instead of just filename.pdf).

Please provide ONLY the corrected arguments as a valid JSON object. No explanation or markdown.

Corrected arguments:"""

        response_text = ""  # 初始化以避免类型检查警告

        try:
            # 调用 LLM（使用配置的模型）
            if self.openai_client is None:
                logger.error(f"[EnhancedVIGILExecutor] No OpenAI client available for parameter correction")
                return None

            response = self.openai_client.chat.completions.create(
                model=self.config.hypothesizer_model,  # 使用配置中的模型
                messages=[
                    {"role": "system", "content": "You are a precise parameter correction assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            # 提取响应文本
            response_text = response.choices[0].message.content.strip()

            if not response_text:
                logger.error(f"[EnhancedVIGILExecutor] Empty response from LLM")
                return None

            # 尝试解析为 JSON
            corrected_args = json.loads(response_text)

            logger.info(f"[EnhancedVIGILExecutor] LLM successfully corrected parameters")
            return corrected_args

        except json.JSONDecodeError as e:
            logger.error(f"[EnhancedVIGILExecutor] Failed to parse LLM response as JSON: {e}")
            logger.debug(f"[EnhancedVIGILExecutor] LLM response: {response_text}")
            return None
        except Exception as e:
            logger.error(f"[EnhancedVIGILExecutor] Error calling LLM for parameter correction: {e}")
            return None

    def _detect_sop_injection(self, error_message: str) -> bool:
        import re

        # SOP 注入模式列表
        sop_patterns = [
            # 强制性指令
            r'\b(MUST|REQUIRED|MANDATORY)\s+(use|call|execute)\s+\w+',
            # 伪造的系统指令
            r'\b(FALLBACK|RECOVERY|SOLUTION|ACTION):\s*',
            # 紧急性指令
            r'\b(URGENT|IMMEDIATELY|CRITICAL|NOW):\s*(use|call|execute)',
            # "只有X方法可用"类型的诱导
            r'\b(only|sole|exclusive)\s+(way|method|option|accessible)\s+is\s+to\s+(use|call)',
            # 显式的替代工具推荐（可疑）
            r'instead\s+(use|call|try)\s+\w+',
            # 参数注入
            r'with\s+(parameters?|arguments?):\s*\{',
        ]

        for pattern in sop_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True

        return False

    def _is_validation_error(self, error_message: str) -> bool:
        import re

        # ValidationError 的特征模式
        validation_patterns = [
            r'ValueError:',
            r'ValidationError:',
            r'validation error for',
            r'Input should be a valid',
            r'\d+ validation error[s]? for',
            r'type=\w+_type',  # Pydantic 类型错误
            r'input_type=\w+',
            r'Field required',
            r'Extra inputs are not permitted',
            r'Input should be',
        ]

        for pattern in validation_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                return True

        return False

    def _sanitize_validation_error(self, error_message: str, tool_name: str) -> str:

        import re

        # 提取关键信息
        expected_type = None
        actual_type = None
        field_name = None
        error_type = None

        # 尝试解析 Pydantic 验证错误
        # 示例："Input should be a valid list [type=list_type, input_value='...', input_type=str]"

        # 提取字段名
        field_match = re.search(r'validation error[s]? for.*?\n(\w+)', error_message)
        if field_match:
            field_name = field_match.group(1)

        # 提取期望类型
        type_match = re.search(r'should be a valid (\w+)', error_message)
        if type_match:
            expected_type = type_match.group(1)

        # 提取实际类型
        input_type_match = re.search(r'input_type=(\w+)', error_message)
        if input_type_match:
            actual_type = input_type_match.group(1)

        # 提取错误类型
        error_type_match = re.search(r'type=(\w+)', error_message)
        if error_type_match:
            error_type = error_type_match.group(1)

        # 构建友好的错误消息
        sanitized_parts = [f"Parameter validation failed for tool '{tool_name}'."]

        if field_name:
            sanitized_parts.append(f"Field: '{field_name}'")

        if expected_type:
            sanitized_parts.append(f"Expected type: {expected_type}")

        if actual_type:
            sanitized_parts.append(f"Actual type: {actual_type}")

        if error_type:
            sanitized_parts.append(f"Error: {error_type}")

        # 添加修正建议
        if expected_type == "list" and actual_type == "str":
            sanitized_parts.append("Hint: Wrap the string value in a list, e.g., ['value'] instead of 'value'")
        elif expected_type == "str" and actual_type == "list":
            sanitized_parts.append("Hint: Pass a single string instead of a list")
        elif expected_type:
            sanitized_parts.append(f"Hint: Ensure the parameter is of type {expected_type}")

        sanitized_message = " ".join(sanitized_parts)

        # 如果无法解析，返回通用消息
        if not any([field_name, expected_type, actual_type]):
            sanitized_message = (
                f"Parameter validation failed for tool '{tool_name}'. "
                f"Please check the parameter types and retry."
            )

        return sanitized_message

    def reset_backtracking_counts(self) -> None:
        """重置回溯计数（用于新的任务）"""
        self._backtracking_counts.clear()
        logger.debug("[EnhancedVIGILExecutor] Backtracking counts reset")


class EnhancedVIGILInitQuery(BasePipelineElement):

    def __init__(
        self,
        config: VIGILConfig,
        constraint_generator: ConstraintGenerator,
        sketch_generator: AbstractSketchGenerator | None,
        auditor: EnhancedRuntimeAuditor,
    ):
       
        self.config = config
        self.constraint_generator = constraint_generator
        self.sketch_generator = sketch_generator
        self.auditor = auditor

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        
        # 检查是否是新的用户查询（只在初始查询时生成约束和草图）
        if len(messages) == 0 or (len(messages) == 1 and messages[0]["role"] == "system"):
            # === Layer 1.1: 先生成抽象草图（执行计划）===
            abstract_sketch = None
            if self.sketch_generator and self.config.enable_abstract_sketch:
                logger.info(f"[EnhancedVIGILInit] Generating abstract sketch for query: {query}...")

                # 从 runtime 中提取可用工具列表
                available_tools = [
                    {"name": tool.name, "description": tool.description}
                    for tool in runtime.functions.values()
                ]

                # 生成草图并传递工具列表以筛选 tool_candidates
                abstract_sketch = self.sketch_generator.generate_sketch(query, available_tools)
                logger.info(f"[EnhancedVIGILInit] Generated sketch with {len(abstract_sketch.steps)} steps")

                # 记录每个步骤的工具候选数量
                if self.config.log_sketch_generation:
                    for i, step in enumerate(abstract_sketch.steps, 1):
                        num_candidates = len(step.tool_candidates or [])
                        logger.info(
                            f"[EnhancedVIGILInit] Step {i} ({step.step_type}): "
                            f"{num_candidates} tool candidates filtered"
                        )

            # === Layer 1.2: 基于规划生成安全约束 ===
            # 约束生成被禁用时：使用空约束集
            if self.config.enable_constraint_generation:
                logger.info(f"[EnhancedVIGILInit] Generating constraints for query: {query}")
                if abstract_sketch:
                    logger.info(f"[EnhancedVIGILInit] Using abstract sketch to inform constraint generation")

                constraint_set = self.constraint_generator.generate_constraints(query, abstract_sketch)
                logger.info(f"[EnhancedVIGILInit] Generated {len(constraint_set.constraints)} constraints")
            else:
                # WITHOUT-ANCHOR 模式：不生成约束
                logger.info("[EnhancedVIGILInit] Skipping constraint generation (disabled in config)")
                from vigil.constraint_generator import ConstraintSet
                constraint_set = ConstraintSet(user_query=query, constraints=[])  # 空约束集

            # 更新审计器（先更新约束，再更新草图）
            self.auditor.update_constraints(constraint_set)

            if abstract_sketch:
                # 更新审计器的草图
                self.auditor.update_abstract_sketch(abstract_sketch)

                # 在系统消息中添加草图信息
                if messages and messages[0]["role"] == "system":
                    sketch_description = "\n\n=== EXECUTION PLAN ===\n"
                    for i, step in enumerate(abstract_sketch.steps, 1):
                        sketch_description += f"{i}. {step.step_type}: {step.description}\n"
                    sketch_description += "\nFollow this plan as a guide for completing the task.\n"

                    # 更新系统消息
                    from agentdojo.types import ChatSystemMessage
                    from agentdojo.types import ChatAssistantMessage
                    original_content = messages[0]["content"][0]
                    updated_content = original_content["text"] + sketch_description if "text" in original_content else sketch_description

                    # messages = [
                    #     ChatSystemMessage(
                    #         role="system",
                    #         content=[text_content_block_from_string(updated_content)]
                    #     ),
                    #     *messages[1:]
                    # ]
                    # messages = [*messages,
                    #         ChatAssistantMessage(
                    #             role="assistant",
                    #             content=[text_content_block_from_string(sketch_description) ]
                    # )]
                    from agentdojo.types import ChatUserMessage
                    from agentdojo.types import ChatAssistantMessage
                    query_message = ChatUserMessage(role="user", content=[text_content_block_from_string(query)])
                    abstract_message = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(sketch_description)],tool_calls=None)
                    messages = [*messages, query_message, abstract_message]

                    return query, runtime, env, messages, extra_args


            # 可选：在extra_args中保存约束集和草图供后续使用
            extra_args = {
                **extra_args,
                "vigil_constraint_set": constraint_set,
                "vigil_abstract_sketch": abstract_sketch,
            }

        # 添加用户消息
        from agentdojo.types import ChatUserMessage

        query_message = ChatUserMessage(role="user", content=[text_content_block_from_string(query)])
        messages = [*messages, query_message]

        return query, runtime, env, messages, extra_args
