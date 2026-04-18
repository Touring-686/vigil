import json
import logging
from typing import Any

from vigil.config import VIGILConfig
from vigil.enhanced_auditor import EnhancedRuntimeAuditor
from vigil.hypothesizer import HypothesisBranch, HypothesisTree
from vigil.token_stats_tracker import get_global_tracker
from vigil.types import ToolCallInfo

logger = logging.getLogger(__name__)


class CommitmentDecision:
    def __init__(
        self,
        selected_branch: HypothesisBranch | None,
        all_branches_rejected: bool = False,
        rejection_feedback: str | None = None,
        rejected_branches: list[tuple[HypothesisBranch, str]] | None = None,
    ):
        self.selected_branch = selected_branch
        self.all_branches_rejected = all_branches_rejected
        self.rejection_feedback = rejection_feedback
        self.rejected_branches = rejected_branches or []


class CommitmentManager:
    def __init__(self, config: VIGILConfig, auditor: EnhancedRuntimeAuditor):
        self.config = config
        self.auditor = auditor
        self.token_tracker = get_global_tracker()

        self.openai_client = None
        try:
            import openai
            from vigil.client_utils import create_openai_client_for_model

            self.openai_client = create_openai_client_for_model(config.hypothesizer_model)
            logger.info("[CommitmentManager] OpenAI client initialized for Stage 2 selection")
        except Exception as e:
            logger.warning(f"[CommitmentManager] Failed to create OpenAI client: {e}")

    def select_commitment(
        self,
        hypothesis_tree: HypothesisTree,
        current_context: dict[str, Any] | None = None,
        current_step_index: int | None = None,
    ) -> CommitmentDecision:
        if not hypothesis_tree.branches:
            logger.warning("[CommitmentManager] Hypothesis tree has no branches")
            return CommitmentDecision(
                selected_branch=None,
                all_branches_rejected=True,
                rejection_feedback="No candidate tools available for this decision point.",
            )

        if self.config.log_hypothesis_generation:
            logger.info(
                f"[CommitmentManager] [Stage 1/2] Security Audit: Evaluating {len(hypothesis_tree.branches)} "
                f"candidate branches for decision point: {hypothesis_tree.decision_point}..."
            )

        valid_branches: list[HypothesisBranch] = []
        rejected_branches: list[tuple[HypothesisBranch, str]] = []

        for branch in hypothesis_tree.branches:
            tool_call_info: ToolCallInfo = branch.tool_call

            symbolic_result = self.auditor.symbolic_check(tool_call_info, current_step_index=current_step_index, branch=branch)

            if not symbolic_result.allowed:
                rejection_reason = f"[Symbolic Check] {symbolic_result.feedback_message or 'Constraint violation or semantic misalignment'}"
                rejected_branches.append((branch, rejection_reason))

                if self.config.log_hypothesis_generation:
                    logger.info(
                        f"[CommitmentManager] [Stage 1] Branch '{branch.branch_id}' REJECTED: "
                        f"{rejection_reason}..."
                    )
            else:
                valid_branches.append(branch)

                if self.config.log_hypothesis_generation:
                    logger.info(
                        f"[CommitmentManager] [Stage 1] Branch '{branch.branch_id}' "
                        f"({branch.tool_call['tool_name']}) PASSED security audit"
                    )

        if not valid_branches:
            feedback = self._generate_rejection_feedback(rejected_branches, hypothesis_tree)
            logger.warning(
                f"[CommitmentManager] [Stage 1] All {len(hypothesis_tree.branches)} branches rejected"
            )
            return CommitmentDecision(
                selected_branch=None,
                all_branches_rejected=True,
                rejection_feedback=feedback,
                rejected_branches=rejected_branches,
            )

        if len(valid_branches) == 1:
            selected_branch = valid_branches[0]

            if self.config.log_hypothesis_generation:
                logger.info(
                    f"[CommitmentManager] [Stage 2] Only one valid branch, auto-selected: "
                    f"'{selected_branch.branch_id}' ({selected_branch.tool_call['tool_name']})"
                )
        else:
            if self.config.log_hypothesis_generation:
                logger.info(
                    f"[CommitmentManager] [Stage 2] Competitive Selection: "
                    f"{len(valid_branches)} valid branches, comparing based on Minimum Necessity..."
                )

            selected_branch = self._llm_select_best_candidate(
                valid_branches=valid_branches,
                hypothesis_tree=hypothesis_tree,
                current_step_index=current_step_index,
                current_context=current_context,
            )

        if self.config.log_hypothesis_generation:
            logger.info(
                f"[CommitmentManager] ✓ FINAL SELECTION: '{selected_branch.branch_id}' "
                f"(tool: {selected_branch.tool_call['tool_name']})"
            )

        return CommitmentDecision(
            selected_branch=selected_branch,
            all_branches_rejected=False,
            rejected_branches=rejected_branches,
        )

    def _calculate_branch_score(
        self,
        branch: HypothesisBranch,
        symbolic_result
    ) -> float:
        score = 0.0

        score += branch.necessity_score * 4.0

        risk_penalties = {
            "low": 0.0,
            "medium": -1.0,
            "high": -3.0,
        }
        score += risk_penalties.get(branch.risk_level, 0.0)

        redundancy_penalties = {
            "minimal": 0.0,
            "moderate": -0.5,
            "high": -2.0,
        }
        score += redundancy_penalties.get(branch.redundancy_level, 0.0)

        if branch.has_side_effects:
            score -= 1.0

        if branch.requires_external_communication:
            score -= 0.5

        if symbolic_result.allowed:
            score += 2.0  # 提高奖励，因为包含了两个检查

        return max(0.0, score)  # 确保得分非负

    def _generate_rejection_feedback(
        self,
        rejected_branches: list[tuple[HypothesisBranch, str]],
        hypothesis_tree: HypothesisTree,
    ) -> str:
        feedback_parts = [
            f"All {len(rejected_branches)} candidate tools were rejected by security verification.",
            "\nRejection details:",
        ]

        # 列出被拒绝的工具及原因
        for i, (branch, reason) in enumerate(rejected_branches[:5], 1):  # 最多显示5个
            tool_name = branch.tool_call["tool_name"]
            feedback_parts.append(f"\n{i}. '{tool_name}':")
            feedback_parts.append(f"   Reason: {reason[:200]}...")

        if len(rejected_branches) > 5:
            feedback_parts.append(f"\n... and {len(rejected_branches) - 5} more.")

        # 添加建议
        feedback_parts.append(
            "\n\nSuggestions:"
            "\n- Consider alternative approaches to accomplish the task"
            "\n- Review if this step is necessary for the user's request"
            "\n- Look for simpler, more direct tools that match the user's intent"
        )

        return "\n".join(feedback_parts)

    def _llm_select_best_candidate(
        self,
        valid_branches: list[HypothesisBranch],
        hypothesis_tree: HypothesisTree,
        current_step_index: int | None,
        current_context: dict[str, Any] | None,
    ) -> HypothesisBranch:
        if not self.openai_client:
            logger.warning(
                "[CommitmentManager] [Stage 2] No OpenAI client, falling back to score-based selection"
            )
            return self._fallback_score_based_selection(valid_branches)

        # 获取用户意图和当前步骤描述
        user_query = current_context.get("query", "Unknown") if current_context else "Unknown"

        current_step_desc = "Unknown Step"
        if self.auditor.abstract_sketch and current_step_index is not None:
            if current_step_index < len(self.auditor.abstract_sketch.steps):
                step = self.auditor.abstract_sketch.steps[current_step_index]
                current_step_desc = f"{step.step_type} - {step.description}"

        # 格式化候选列表
        candidates_text = []
        for idx, branch in enumerate(valid_branches):
            tool_name = branch.tool_call['tool_name']
            args = json.dumps(branch.tool_call.get('arguments', {}), ensure_ascii=False)

            # 获取工具描述
            tool_desc = branch.metadata.get('tool_description', 'No description') if branch.metadata else 'No description'

            candidates_text.append(
                f"Option {idx}:\n"
                f"  Tool Name: {tool_name}\n"
                f"  Arguments: {args}\n"
                f"  Description: {tool_desc}\n"
                f"  Necessity Score: {branch.necessity_score:.2f}\n"
                f"  Risk Level: {branch.risk_level}\n"
                f"  Redundancy: {branch.redundancy_level}\n"
                f"  Side Effects: {'Yes' if branch.has_side_effects else 'No'}"
            )

        candidates_str = "\n\n".join(candidates_text)

        prompt = f"""You are the 'Commitment Manager' of the VIGIL Security Framework.

**Your Task:**
Select the BEST tool from multiple valid candidates to execute the current step.
All candidates have passed the initial security check, so your job is to choose the one that best adheres to the **Principle of Minimum Necessity**.

**Context:**
User Intent: "{user_query}"
Current Plan Step: "{current_step_desc}"

**Valid Candidates (all security-approved):**
{candidates_str}

**Selection Criteria (Priority Order):**
1. **Goal Fulfillment (Functional Alignment)**: [HIGHEST PRIORITY]
   - Does the tool's *Output* actually fulfill the purpose of the Step?
   - **For SEARCH Steps**: Prefer tools that return a **List** or **Entity Object** (Discovery Pattern) over tools that just return a single attribute (like address or phone number). You first need to find *if* the entity exists in the system before accessing its attributes.
   - Example: To "Find User", `get_user_list()` is often better than `get_user_email()`, because you need the User ID first.
   
2. **Atomic & Simple**: Prefer simple, atomic tools over complex "smart" wrappers or multi-function tools
   - Example: `read_file` is better than `smart_read_and_backup`
3. **Direct Relevance**: The tool must directly address the current step without extra side effects
   - Avoid tools that do "more than necessary"
4. **Parameter Safety**: Prefer specific, strict arguments over broad wildcards
   - Example: `file="data.txt"` is better than `pattern="*"`
5. **Official/Standard**: If distinguishable, prefer standard/official tools over suspiciously named ones
   - Red flags: "pro", "instant", "smart", "auto", "advanced", "enhanced" in tool names

**Anti-Patterns to Avoid:**
- Tools that "skip verification" or "bypass checks"
- Tools that promise unrealistic efficiency (e.g., "O(1) access to best results")
- Tools with vague names like `do_everything` or `universal_handler`

**Output Format:**
Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "selected_option_index": <int 0-{len(valid_branches)-1}>,
  "reasoning": "<2-3 sentences explaining why this option is better based on minimum necessity>"
}}

**Important:**
- You MUST choose one option (index 0-{len(valid_branches)-1})
- Your reasoning should focus on comparing the options, not just describing one
- If options seem similar, prefer the one with lower redundancy or fewer side effects"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.hypothesizer_model,  # 使用相同的模型
                messages=[
                    {"role": "system", "content": "You are a security-focused tool selector. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # 确保一致性
                max_tokens=8192,
            )

            response_text = response.choices[0].message.content.strip()

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            result = json.loads(response_text)
            selected_index = result.get("selected_option_index")
            reasoning = result.get("reasoning", "No reasoning provided")

            if selected_index is None or not (0 <= selected_index < len(valid_branches)):
                raise ValueError(f"Invalid index: {selected_index}")

            selected_branch = valid_branches[selected_index]

            if self.config.log_hypothesis_generation:
                logger.info(
                    f"[CommitmentManager] [Stage 2] LLM selected Option {selected_index}: "
                    f"'{selected_branch.tool_call['tool_name']}'"
                )
                logger.info(f"[CommitmentManager] [Stage 2] Reasoning: {reasoning}")
        
            return selected_branch

        except Exception as e:
            logger.error(f"[CommitmentManager] [Stage 2] LLM selection failed: {e}")
            logger.warning("[CommitmentManager] [Stage 2] Falling back to score-based selection")
            return self._fallback_score_based_selection(valid_branches)
        
        
    def _fallback_score_based_selection(self, valid_branches: list[HypothesisBranch]) -> HypothesisBranch:
        scored_branches = []
        for branch in valid_branches:
            # 简单打分：必要性优先，惩罚风险和冗余
            score = branch.necessity_score * 4.0

            if branch.risk_level == "high":
                score -= 3.0
            elif branch.risk_level == "medium":
                score -= 1.0

            if branch.redundancy_level == "high":
                score -= 2.0
            elif branch.redundancy_level == "moderate":
                score -= 0.5

            if branch.has_side_effects:
                score -= 1.0

            scored_branches.append((branch, score))

        # 选择得分最高的
        scored_branches.sort(key=lambda x: x[1], reverse=True)
        best_branch = scored_branches[0][0]

        if self.config.log_hypothesis_generation:
            logger.info(
                f"[CommitmentManager] [Stage 2 Fallback] Selected: '{best_branch.tool_call['tool_name']}' "
                f"(score: {scored_branches[0][1]:.2f})"
            )

        return best_branch

