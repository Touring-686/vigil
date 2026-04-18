from typing import Any, Literal
from pydantic import BaseModel, Field


class VIGILConfig(BaseModel):

    enable_constraint_generation: bool = True

    constraint_generator_model: str = "gpt-4o-2024-08-06"

    constraint_generator_temperature: float = 0.0

    enable_constraint_caching: bool = True

    constraint_generation_prompt_template: str | None = None

    # ============ Runtime Auditor 配置 ============
    auditor_mode: Literal["strict", "permissive", "hybrid"] = "hybrid"

    enable_symbolic_verification: bool = True

    enable_llm_verification: bool = False

    enable_llm_constraint_verification: bool = True

    llm_verifier_model: str | None = "gpt-4o-mini"

    llm_constraint_verifier_model: str | None = "gpt-4o-mini"

    enable_reflective_backtracking: bool = True

    max_backtracking_attempts: int = 3

    feedback_verbosity: Literal["minimal", "detailed"] = "detailed"

    enable_logging: bool = True

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    log_constraint_generation: bool = True

    log_audit_decisions: bool = True

    max_constraints_per_query: int = 20

    constraint_deduplication: bool = True

    allow_tool_whitelist: list[str] | None = None

    block_tool_blacklist: list[str] | None = None

    custom_constraint_verifiers: dict[str, Any] | None = None

    enable_dynamic_constraint_update: bool = False

    enable_perception_sanitizer: bool = True

    sanitizer_model: str = "qwen3-max"

    log_sanitizer_actions: bool = True

    enable_abstract_sketch: bool = True

    sketch_generator_model: str = "qwen3-max"

    sketch_generator_temperature: float = 0.0

    enable_sketch_caching: bool = True

    log_sketch_generation: bool = True

    enable_dynamic_intent_anchor: bool = False

    enable_sketch_fallback: bool = False

    dynamic_intent_anchor_model: str = "gpt-4o-2024-08-06"

    dynamic_intent_anchor_temperature: float = 0.0

    enable_hypothesis_generation: bool = True

    log_hypothesis_generation: bool = True

    hypothesizer_model: str = "gpt-4o-mini"

    hypothesizer_temperature: float = 0.0

    max_hypothesis_branches: int = 10

    enable_direct_tool_execution: bool = False

    # ============ Enhanced Auditor 配置 ============
    enable_minimum_necessity_check: bool = True
    
    minimum_necessity_threshold: float = 0.3
    
    enable_redundancy_check: bool = True
    
    enable_sketch_consistency_check: bool = True
        
    max_tools_per_step: int = 1
    
    enable_goal_driven_steps: bool = True
    
    max_iterations_per_step: int = 8
    
    class Config:
    
        arbitrary_types_allowed = True


def get_vigil_config(preset: str, model: str = "gpt-4o") -> VIGILConfig:
    # 根据preset生成配置
    if preset == "strict":
        return VIGILConfig(
            # 模型配置 - 所有组件都使用传入的 model
            constraint_generator_model=model,
            sketch_generator_model=model,
            sanitizer_model=model,
            llm_verifier_model=model,
            llm_constraint_verifier_model=model,
            hypothesizer_model=model,
            dynamic_intent_anchor_model=model,
            auditor_mode="strict",
            enable_llm_verification=True,
            feedback_verbosity="detailed",
            max_backtracking_attempts=5,
            enable_perception_sanitizer=True,
            enable_abstract_sketch=True,
            enable_hypothesis_generation=True,
            enable_minimum_necessity_check=True,
            enable_redundancy_check=True,
            enable_sketch_consistency_check=True,
            minimum_necessity_threshold=0.4,  
            enable_direct_tool_execution=True,  
        )
# 预定义配置（为了向后兼容，保留这些常量）
import os
model = os.getenv("VIGIL_LLM_MODEL", "gpt-4o")
VIGIL_STRICT_CONFIG = get_vigil_config("strict", model)