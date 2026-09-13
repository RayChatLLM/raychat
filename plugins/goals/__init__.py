"""Register persistent goals with complete-transcript judging and typed services."""

from raychat.service_contracts import GoalCommand as GoalCommand
from raychat.service_contracts import GoalDecision as GoalDecision
from raychat.service_contracts import GoalStatus as GoalStatus
from raychat.service_contracts import JudgedTurn as JudgedTurn
from raychat.service_contracts import JudgeProfile as JudgeProfile
from raychat.service_contracts import JudgeRouter as JudgeRouter
from raychat.service_contracts import JudgeSession as JudgeSession

from .commands import parse_goal_command as parse_goal_command
from .controller import GoalController as GoalController
from .judge import JUDGE_INSTRUCTIONS as JUDGE_INSTRUCTIONS
from .judge import GoalJudge as GoalJudge
from .judge import GoalJudgeResponseError as GoalJudgeResponseError
from .registration import register as register
