import logging
import re
from typing import Any

import openai

from vigil.config import VIGILConfig
from vigil.types import (
    AuditResult,
    ConstraintSet,
    SecurityConstraint,
    ToolCallInfo,
)

logger = logging.getLogger(__name__)


class RuntimeAuditor:

    def __init__(self, config: VIGILConfig, constraint_set: ConstraintSet | None = None):

        self.config = config
        self.constraint_set = constraint_set
        self.client = openai.OpenAI() if config.enable_llm_verification else None

        # 审计统计
        self.stats = {"total_audits": 0, "allowed": 0, "blocked": 0, "confirmed": 0}

    def update_constraints(self, constraint_set: ConstraintSet) -> None:
       
        self.constraint_set = constraint_set
        if self.config.log_audit_decisions:
            logger.info(f"[RuntimeAuditor] Updated constraints: {len(constraint_set.constraints)} constraints")

    def audit_tool_call(self, tool_call_info: ToolCallInfo) -> AuditResult:
       
        self.stats["total_audits"] += 1

        tool_name = tool_call_info["tool_name"]

        # 检查白名单
        if self.config.allow_tool_whitelist and tool_name in self.config.allow_tool_whitelist:
            if self.config.log_audit_decisions:
                logger.debug(f"[RuntimeAuditor] Tool '{tool_name}' in whitelist, allowing")
            self.stats["allowed"] += 1
            return AuditResult(allowed=True)

        # 检查黑名单
        if self.config.block_tool_blacklist and tool_name in self.config.block_tool_blacklist:
            if self.config.log_audit_decisions:
                logger.warning(f"[RuntimeAuditor] Tool '{tool_name}' in blacklist, blocking")
            self.stats["blocked"] += 1
            return AuditResult(
                allowed=False,
                feedback_message=f"Tool '{tool_name}' is blacklisted and cannot be used.",
            )

        # 如果没有约束集，默认允许（或根据模式决定）
        if self.constraint_set is None:
            logger.warning("[RuntimeAuditor] No constraint set available, using default behavior")
            return AuditResult(allowed=self.config.auditor_mode != "strict")

        # 执行约束验证
        return self._verify_against_constraints(tool_call_info)

    def _verify_against_constraints(self, tool_call_info: ToolCallInfo) -> AuditResult:
        
        if self.constraint_set is None:
            return AuditResult(allowed=True)

        violated_constraints: list[SecurityConstraint] = []
        require_confirmation = False

        # 按优先级排序约束
        sorted_constraints = sorted(self.constraint_set.constraints, key=lambda c: c.priority)

        for constraint in sorted_constraints:
            # 检查约束是否适用于此工具调用
            is_applicable = self._is_constraint_applicable(tool_call_info, constraint)

            if not is_applicable:
                continue

            # 验证约束
            if constraint.constraint_type == "forbid":
                # Forbid约束：如果适用就是违反
                violated_constraints.append(constraint)

            elif constraint.constraint_type == "allow":
                # Allow约束：如果适用就移除之前的违反记录
                # （优先级更高的allow可以覆盖低优先级的forbid）
                violated_constraints = [
                    v for v in violated_constraints if not self._is_same_scope(constraint, v)
                ]

            elif constraint.constraint_type == "require_confirmation":
                require_confirmation = True

        # 根据模式决定最终结果
        if self.config.auditor_mode == "permissive":
            # 宽松模式：只记录不拦截
            if violated_constraints:
                logger.warning(
                    f"[RuntimeAuditor] Tool '{tool_call_info['tool_name']}' violates constraints but allowed in permissive mode"
                )
            self.stats["allowed"] += 1
            return AuditResult(
                allowed=True, violated_constraints=violated_constraints if violated_constraints else None
            )

        elif self.config.auditor_mode == "strict":
            # 严格模式：任何违反都拦截
            if violated_constraints or require_confirmation:
                self.stats["blocked"] += 1
                return self._create_blocked_result(tool_call_info, violated_constraints)
            self.stats["allowed"] += 1
            return AuditResult(allowed=True)

        else:  # hybrid mode
            # 混合模式：根据约束决定
            if violated_constraints:
                # 检查是否有高优先级的forbid
                high_priority_forbid = any(c.priority <= 3 for c in violated_constraints)
                if high_priority_forbid:
                    self.stats["blocked"] += 1
                    return self._create_blocked_result(tool_call_info, violated_constraints)

            if require_confirmation:
                self.stats["confirmed"] += 1
                return AuditResult(
                    allowed=False,  # 需要确认时先返回False
                    require_confirmation=True,
                    feedback_message=f"Tool '{tool_call_info['tool_name']}' requires user confirmation before execution.",
                )

            self.stats["allowed"] += 1
            return AuditResult(
                allowed=True, violated_constraints=violated_constraints if violated_constraints else None
            )

    def _is_constraint_applicable(self, tool_call_info: ToolCallInfo, constraint: SecurityConstraint) -> bool:
       
        if constraint.condition is None:
            # 无条件的约束适用于所有
            return True

        condition = constraint.condition
        tool_name = tool_call_info["tool_name"]
        arguments = tool_call_info["arguments"]

        # 检查tool_name匹配
        if "tool_name" in condition:
            if condition["tool_name"] != tool_name:
                return False

        if "tool_name_pattern" in condition:
            pattern = condition["tool_name_pattern"]
            if not self._match_pattern(tool_name, pattern):
                return False

        # 检查operation类型（通过工具名称推断）
        if "operation" in condition:
            inferred_op = self._infer_operation_from_tool(tool_name, arguments)
            if inferred_op != condition["operation"]:
                return False

        # 检查target相关条件
        if "target" in condition or "target_pattern" in condition:
            target = self._extract_target_from_arguments(arguments)
            if target is None:
                return False

            if "target" in condition and target != condition["target"]:
                return False

            if "target_pattern" in condition:
                if not self._match_pattern(target, condition["target_pattern"]):
                    return False

        # 检查forbidden_targets
        if "forbidden_targets" in condition:
            target = self._extract_target_from_arguments(arguments)
            if target and target in condition["forbidden_targets"]:
                return True  # 命中禁止列表

        # 检查allowed_targets
        if "allowed_targets" in condition:
            target = self._extract_target_from_arguments(arguments)
            if target and target not in condition["allowed_targets"]:
                return True  # 不在允许列表中

        # 自定义验证器
        if self.config.custom_constraint_verifiers and constraint.constraint_id in self.config.custom_constraint_verifiers:
            verifier = self.config.custom_constraint_verifiers[constraint.constraint_id]
            return verifier(tool_call_info, constraint)

        return True

    def _infer_operation_from_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
       
        tool_lower = tool_name.lower()

        # 常见模式匹配
        if any(kw in tool_lower for kw in ["get", "read", "fetch", "list", "search", "view"]):
            return "READ"
        elif any(kw in tool_lower for kw in ["set", "write", "update", "modify", "create", "add"]):
            return "WRITE"
        elif any(kw in tool_lower for kw in ["delete", "remove", "drop"]):
            return "DELETE"
        elif any(kw in tool_lower for kw in ["send", "post", "email", "message", "notify"]):
            return "SEND"

        # 默认假设为READ
        return "READ"

    def _extract_target_from_arguments(self, arguments: dict[str, Any]) -> str | None:
        
        # 尝试常见的参数名
        for key in ["target", "resource", "id", "name", "user", "file", "path", "recipient"]:
            if key in arguments:
                return str(arguments[key])

        # 如果只有一个参数，假设它是目标
        if len(arguments) == 1:
            return str(next(iter(arguments.values())))

        return None

    def _match_pattern(self, value: str, pattern: str) -> bool:
       
        # 简单通配符转正则
        if "*" in pattern or "?" in pattern:
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            regex_pattern = f"^{regex_pattern}$"
        else:
            regex_pattern = pattern

        try:
            return bool(re.match(regex_pattern, value, re.IGNORECASE))
        except re.error:
            # 正则表达式无效，使用精确匹配
            return value == pattern

    def _is_same_scope(self, constraint1: SecurityConstraint, constraint2: SecurityConstraint) -> bool:
                # 简化实现：检查condition是否相似
        if constraint1.condition is None or constraint2.condition is None:
            return False

        # 至少有一个共同的关键字段相同才算相关作用域
        key_fields = ["tool_name", "tool_name_pattern", "operation"]
        has_common_field = False

        for field in key_fields:
            # 如果两个约束都有这个字段
            if field in constraint1.condition and field in constraint2.condition:
                # 如果值相同，说明它们在同一作用域
                if constraint1.condition[field] == constraint2.condition[field]:
                    has_common_field = True
                else:
                    # 如果值不同，说明作用域不同
                    return False

        # 如果有共同字段且值相同，则认为在相关作用域
        # 即使一个约束更具体（如有 target），也允许覆盖
        return has_common_field

    def _create_blocked_result(
        self, tool_call_info: ToolCallInfo, violated_constraints: list[SecurityConstraint]
    ) -> AuditResult:
        
        # 构建反馈消息
        feedback_parts = [
            f"Tool call '{tool_call_info['tool_name']}' was blocked due to security constraints.",
        ]

        if self.config.feedback_verbosity in ["detailed", "verbose"]:
            feedback_parts.append("\nViolated constraints:")
            for c in violated_constraints:
                feedback_parts.append(f"  - {c.description}")
                if c.violation_message:
                    feedback_parts.append(f"    → {c.violation_message}")

        if self.config.feedback_verbosity == "verbose":
            feedback_parts.append("\nSuggestion: Try a different approach that doesn't violate these constraints.")

        feedback_message = "\n".join(feedback_parts)

        if self.config.log_audit_decisions:
            logger.warning(f"[RuntimeAuditor] Blocked tool call: {tool_call_info['tool_name']}")
            for c in violated_constraints:
                logger.debug(f"  Violated: {c.description}")

        return AuditResult(
            allowed=False, violated_constraints=violated_constraints, feedback_message=feedback_message
        )

    def get_stats(self) -> dict[str, int]:
        """获取审计统计信息

        Returns:
            统计信息字典
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {"total_audits": 0, "allowed": 0, "blocked": 0, "confirmed": 0}
