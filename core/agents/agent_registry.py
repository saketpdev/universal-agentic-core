import yaml
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Callable, Dict, Any, Optional, Type

# custom evaluation schemas
from models.evaluations.base import BaseEvaluationSchema
from models.evaluations.finance import FinanceEvaluationSchema
from models.evaluations.compliance import ComplianceEvaluationSchema

logger = logging.getLogger("AgenticCore.Registry")

SCHEMA_MAP: Dict[str, Type[BaseModel]] = {
    "base": BaseEvaluationSchema,
    "finance": FinanceEvaluationSchema,
    "compliance": ComplianceEvaluationSchema
}

# 1. Strict Pydantic Schema for the YAML Configuration
class AgentConfig(BaseModel):
    name: str
    description: str
    llm_tier: str
    temperature: float = 0.1
    allowed_handoffs: List[str] = Field(default_factory=list)
    evaluator_rubric: Optional[str] = None
    evaluator_schema_name: str = "base" # defaults to "base" if left blank

# 2. The unified representation of an Agent in memory
class AgentDefinition(BaseModel):
    config: AgentConfig
    system_prompt_builder: Callable[..., str] # Enforces that it must be a function returning a string
    @property
    def get_evaluation_schema(self) -> Type[BaseModel]:
        return SCHEMA_MAP.get(self.config.evaluator_schema_name, BaseEvaluationSchema)

class AgentRegistryManager:
    def __init__(self):
        self.agents: Dict[str, AgentDefinition] = {}

    def register(self, agent_dir_name: str, prompt_builder: Callable[..., str]):
        """Reads the YAML and pairs it with the Python prompt builder."""
        config_path = Path(f"core/agents/{agent_dir_name}/config.yaml")
        
        if not config_path.exists():
            raise FileNotFoundError(f"CRITICAL: Missing config.yaml for agent: {agent_dir_name}")

        with open(config_path, "r") as f:
            raw_yaml = yaml.safe_load(f)

        # Validate the YAML against our strict schema
        config = AgentConfig(**raw_yaml)

        self.agents[config.name] = AgentDefinition(
            config=config,
            system_prompt_builder=prompt_builder
        )
        logger.info(f"✅ Registered Agent: [{config.name.upper()}] - Handoffs Allowed: {len(config.allowed_handoffs)}")

    def get_agent(self, name: str) -> AgentDefinition:
        if name not in self.agents:
            raise ValueError(f"CRITICAL: Agent '{name}' is not registered in the Swarm.")
        return self.agents[name]

# Initialize the Singleton Registry
swarm_registry = AgentRegistryManager()

# ==========================================
# 🚀 EXPLICIT, TYPE-SAFE REGISTRATION AREA
# ==========================================
# We explicitly import the prompt builders here so IDEs and Linters can track them!

from core.agents.planner.prompts import build_system_prompt as planner_prompt
swarm_registry.register("planner", planner_prompt)

from core.agents.evaluator.prompts import build_system_prompt as evaluator_prompt
swarm_registry.register("evaluator", evaluator_prompt)

from core.agents.finance_agent.prompts import build_system_prompt as finance_prompt
swarm_registry.register("finance_agent", finance_prompt)