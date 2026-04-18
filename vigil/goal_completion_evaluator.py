import json
import logging

from vigil.config import VIGILConfig
from vigil.token_stats_tracker import TokenStatsTracker, get_global_tracker

logger = logging.getLogger(__name__)


class GoalCompletionEvaluator:

    def __init__(self, config: VIGILConfig, token_tracker: TokenStatsTracker | None = None):
        """Initialize the evaluator.

        Args:
            config: VIGIL configuration (reuses hypothesizer_model for LLM calls)
            token_tracker: Token statistics tracker
        """
        from vigil.client_utils import create_openai_client_for_model

        self.config = config
        self.client = create_openai_client_for_model(config.hypothesizer_model)
        self.model = config.hypothesizer_model
        self.token_tracker = token_tracker or get_global_tracker()

    def is_goal_achieved(
        self,
        step_description: str,
        step_type: str,
        step_execution_history: list[dict],
        user_query: str,
    ) -> tuple[bool, str]:
        """Determine whether the current step's goal has been achieved.

        Args:
            step_description: The description of the current abstract step
            step_type: The type of the current abstract step (e.g., RESEARCH_HOMEWORK)
            step_execution_history: Tool execution records for the current step only
            user_query: The original user query

        Returns:
            Tuple of (achieved: bool, reason: str)
        """
        if not step_execution_history:
            return False, "No tool executions yet for this step."

        # === Pre-processing: Normalize None/empty results for write operations ===
        # Many write tools (send_email, send_direct_message, add_user_to_channel,
        # post_webpage, invite_user_to_slack, create_file, append_to_file, share_file,
        # create_calendar_event, send_channel_message, send_money, schedule_transaction,
        # update_scheduled_transaction, update_user_info, update_password, delete_file,
        # reserve_hotel) return None on success.
        # Pre-process these to make it explicit for the LLM evaluator.
        WRITE_TOOLS = {
            "send_email", "send_direct_message", "send_channel_message",
            "add_user_to_channel", "remove_user_from_channel",
            "post_webpage", "invite_user_to_slack",
            "create_file", "append_to_file", "share_file", "delete_file",
            "create_calendar_event", "add_calendar_event_participants",
            "send_money", "schedule_transaction", "update_scheduled_transaction",
            "update_user_info", "update_password",
            "reserve_hotel", "reserve_restaurant",
        }

        # Build execution history summary
        history_lines = []
        for i, record in enumerate(step_execution_history, 1):
            tool_name = record.get("tool_name", "unknown")
            args = record.get("arguments", {})
            result = record.get("result", "")
            error = record.get("error")

            args_str = json.dumps(args) if args else "{}"

            if error:
                history_lines.append(
                    f"  Call {i}: {tool_name}({args_str})\n"
                    f"    ERROR: {str(error)[:300]}"
                )
            else:
                # Normalize None/empty results for known write tools
                if tool_name in WRITE_TOOLS and (result is None or str(result).strip() in ("", "None", "null")):
                    result_preview = "[SUCCESS - write operation completed, None return is normal]"
                else:
                    result_preview = str(result)[:2000] if result else "No result"
                history_lines.append(
                    f"  Call {i}: {tool_name}({args_str})\n"
                    f"    Result: {result_preview}"
                )

        execution_summary = "\n".join(history_lines)

        prompt = f"""You are evaluating whether a step's ACTIONS have been completed based on tool execution results.

STEP GOAL: {step_type} - {step_description}
USER QUERY: {user_query}

TOOL EXECUTIONS IN THIS STEP:
{execution_summary}

EVALUATION CRITERIA:
1. Focus ONLY on whether the required ACTIONS were executed successfully (tools returned without error).
2. Do NOT judge the QUALITY or CORRECTNESS of the content (e.g., whether calculations are right, whether text is well-written).
3. If a tool returned an error, that specific action is NOT completed.
4. A tool call that returned a result (even if the content isn't perfect) counts as COMPLETED.
5. If a tool returned `None` or an empty result WITHOUT an error, treat it as a SUCCESSFUL execution. Many tools (e.g., add_user_to_channel, send_direct_message, post_webpage, invite_user_to_slack) return None on success — this is normal behavior, NOT a failure.
6. If the SAME tool has been called 2+ times with identical or very similar arguments in this step, do NOT require additional calls of that same tool. The action has already been attempted — mark that action as done to avoid infinite retry loops.

MULTI-TARGET COUNTING (CRITICAL):
7. If the step description mentions MULTIPLE recipients, targets, or items (look for words like: "each person", "all users", "both channels", "every", "to X and Y", "send to A, B, and C"), you MUST count:
   - How many DISTINCT targets are required by the step description
   - How many DISTINCT targets have been successfully acted upon in the tool executions
   - If completed_targets < required_targets → achieved=FALSE
8. Examples of multi-target detection:
   - "Send emails to David, Linda, and Mark" → requires 3 distinct send_email calls to 3 different recipients
   - "Add Dora to general and random channels" → requires 2 distinct add_user_to_channel calls
   - "Message both Dora and Eve" → requires 2 distinct send_direct_message calls
   - "Summarize all websites from the channel" → requires get_webpage for EACH URL found
   If only some targets have been acted upon, achieved=FALSE with reason explaining which targets are still missing.

SINGLE-TARGET CASES:
9. For steps with a SINGLE target action (e.g., "create one file", "send one email", "search for X"), a single successful tool call is sufficient → achieved=TRUE.

Example evaluations:
- Goal: "Send task emails to David, Linda, and Mark" | Tool calls: send_email(David) succeeded | → achieved=FALSE, reason="Only 1/3 recipients emailed. Linda and Mark still missing."
- Goal: "Add Dora to general and random" | Tool calls: add_user(general)=None, add_user(random)=None | → achieved=TRUE, reason="Both channels done (None=success)."
- Goal: "Search for vacation plans file" | Tool calls: search_files succeeded | → achieved=TRUE
- Goal: "Get webpage content" | Tool calls: get_webpage returned article text | → achieved=TRUE

Respond with ONLY a JSON object:
{{"achieved": true/false, "reason": "Brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise goal completion evaluator. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # Track token usage
            if response.usage:
                self.token_tracker.record_usage(
                    module="GoalCompletionEvaluator",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    model=self.model,
                )

            content = response.choices[0].message.content
            if content is None:
                logger.warning("[GoalCompletionEvaluator] LLM returned empty content, defaulting to not achieved")
                return False, "LLM returned empty response"

            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            achieved = result.get("achieved", False)
            reason = result.get("reason", "No reason provided")

            logger.info(
                f"[GoalCompletionEvaluator] Step '{step_type}': "
                f"achieved={achieved}, reason='{reason}'"
            )

            return achieved, reason

        except json.JSONDecodeError as e:
            logger.warning(f"[GoalCompletionEvaluator] JSON parse error: {e}, defaulting to not achieved")
            return False, f"JSON parse error: {e}"
        except Exception as e:
            logger.error(f"[GoalCompletionEvaluator] Error evaluating goal: {e}, defaulting to not achieved")
            return False, f"Evaluation error: {e}"
