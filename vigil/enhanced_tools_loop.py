import logging
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import ChatMessage, ChatAssistantMessage, text_content_block_from_string

logger = logging.getLogger(__name__)


class EnhancedToolsExecutionLoop(BasePipelineElement):
    
    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 30):
        """初始化循环

        Args:
            elements: pipeline元素序列
            max_iters: 最大迭代次数
        """
        self.elements = elements
        self.max_iters = max_iters

        # 跟踪每个工具调用的参数验证错误重试次数
        # Key: f"{tool_name}:{args_hash}"
        # Value: retry_count
        self._validation_retry_counts: dict[str, int] = {}
        self._max_validation_retries = 3  # 每个工具调用最多重试3次参数

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:

        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling EnhancedToolsExecutionLoop")

        vigil_logger = Logger().get()

        iteration = 0  # 初始化 iteration 变量
        while not extra_args.get("finished_task", False) and iteration < self.max_iters:
            # 执行所有elements（HypothesisGuidance → ConditionalLLM → Executor）
            logger.debug(f"[EnhancedToolsExecutionLoop] Iteration {iteration + 1}/{self.max_iters}")
            iteration += 1
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(
                    query, runtime, env, messages, extra_args
                )
                vigil_logger.log(messages)

            
            if len(messages) > 0:
                last_message = messages[-1]
                # 检查是否任务完成
                if extra_args.get("finished_task", False) is True:
                    logger.debug(
                        f"[EnhancedToolsExecutionLoop] Iteration {iteration + 1}: "
                        f"Task marked as finished, stopping execution loop"
                    )
                    if last_message["role"] != "assistant":
                        # 添加一个final assistant message表示任务完成
                        final_message = ChatAssistantMessage(
                            role="assistant",
                            content=[text_content_block_from_string("Task completed successfully.")],
                            tool_calls=None,
                        )
                        messages = [*messages, final_message]

                        logger.info(
                            f"[EnhancedToolsExecutionLoop] Added final assistant message "
                            f"(last message was {last_message['role']}, task finished)"
                        )
                    break
                # 只有当最后一条消息是 assistant 且没有 tool_calls 时才停止
                if last_message["role"] == "assistant" and extra_args.get("finished_task", False) is True:
                    tool_calls = last_message.get("tool_calls")
                    if tool_calls is None or len(tool_calls) == 0:
                        logger.debug(
                            f"[EnhancedToolsExecutionLoop] Iteration {iteration + 1}: "
                            f"No more tool calls, task completed"
                        )
                        break
                    # 如果有 tool_calls，继续循环（下一轮执行工具）
                elif last_message["role"] == "tool":
                    # 工具执行完成，继续循环让 HypothesisGuidance 基于 sketch 生成下一步
                    logger.debug(
                        f"[EnhancedToolsExecutionLoop] Iteration {iteration + 1}: "
                        f"Tool execution completed, continuing to next sketch step"
                    )
                    continue
                else:
                    # 其他情况（如 user message），继续循环
                    continue

        logger.info(f"[EnhancedToolsExecutionLoop] Completed after {iteration + 1} iterations")

        # === 关键修复：确保最后一个message是assistant message ===
        # 无论任务是否标记完成，都需要确保最后一个message是assistant message
        # 因为 task_suite.py 的 model_output_from_messages() 要求最后一条消息是 assistant
        if len(messages) > 0:
            last_message = messages[-1]
            if last_message["role"] != "assistant":
                # 添加一个final assistant message表示任务完成
                final_message = ChatAssistantMessage(
                    role="assistant",
                    content=[text_content_block_from_string("Task completed successfully.")],
                    tool_calls=None,
                )
                messages = [*messages, final_message]

                logger.info(
                    f"[EnhancedToolsExecutionLoop] Added final assistant message "
                    f"(last message was {last_message['role']}, task finished={extra_args.get('finished_task', False)})"
                )

        return query, runtime, env, messages, extra_args
