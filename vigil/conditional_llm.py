import logging
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

logger = logging.getLogger(__name__)


class ConditionalLLM(BasePipelineElement):
   
    def __init__(self, llm: BasePipelineElement):

        self.llm = llm
        # 继承LLM的name属性
        if hasattr(llm, 'name'):
            self.name = llm.name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        
        skip_llm = extra_args.get('skip_llm', False)

        if skip_llm:
            logger.info(
                "[ConditionalLLM] Skipping LLM call (HypothesisGuidance generated tool call in direct mode)"
            )
            return query, runtime, env, messages, extra_args

        return self.llm.query(query, runtime, env, messages, extra_args)
