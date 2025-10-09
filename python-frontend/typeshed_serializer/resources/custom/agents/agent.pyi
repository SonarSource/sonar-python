from typing import Any

from SonarPythonAnalyzerFakeStub import CustomStubBase
from agents.run_context import TContext
from agents.handoffs import Handoff
from agents.guardrail import InputGuardrail, OutputGuardrail


class Agent(CustomStubBase):
    handoffs: list[Agent[Any] | Handoff[TContext, Any]]
    input_guardrails: list[InputGuardrail[TContext]]
    output_guardrails: list[OutputGuardrail[TContext]]

