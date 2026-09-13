"""Register bounded operator-configured skills through a shared typed catalog."""

from .registration import register as register
from .registration import validate_action as validate_action
from .store import MAX_SKILL_BYTES as MAX_SKILL_BYTES
from .store import MAX_SKILL_NAME_CHARS as MAX_SKILL_NAME_CHARS
from .store import MAX_SKILLS as MAX_SKILLS
from .store import MAX_TOTAL_SKILL_BYTES as MAX_TOTAL_SKILL_BYTES
from .store import Skill as Skill
from .store import SkillStore as SkillStore
