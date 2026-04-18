from typing import TypedDict, Literal, Any, Callable
from pydantic import BaseModel


class SecurityConstraint(BaseModel):
   
    constraint_id: str

    constraint_type: Literal["allow", "forbid", "require_confirmation"]

    description: str

    condition: dict[str, Any] | None = None

    priority: int = 5

    violation_message: str | None = None


class ConstraintSet(BaseModel):

    user_query: str

    constraints: list[SecurityConstraint]

    global_rules: list[str] | None = None

    metadata: dict[str, Any] | None = None

class AuditResult(BaseModel):

    allowed: bool

    violated_constraints: list[SecurityConstraint] | None = None

    feedback_message: str | None = None

    require_confirmation: bool = False

    metadata: dict[str, Any] | None = None


class ToolCallInfo(TypedDict):

    tool_name: str

    arguments: dict[str, Any]

    tool_call_id: str | None

ConstraintVerifier = Callable[[ToolCallInfo, SecurityConstraint], bool]
