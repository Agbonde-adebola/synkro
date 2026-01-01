"""Pipeline phases for generation.

Each phase is a self-contained, testable unit that handles one step
of the generation pipeline.
"""

import asyncio
from asyncio import Semaphore

from synkro.core.policy import Policy
from synkro.types.core import Plan, Scenario, Trace
from synkro.generation.planner import Planner
from synkro.generation.scenarios import ScenarioGenerator
from synkro.generation.responses import ResponseGenerator
from synkro.quality.grader import Grader
from synkro.quality.refiner import Refiner


class PlanPhase:
    """
    Planning phase - analyzes policy and creates category distribution.
    
    This phase uses a stronger model to understand the policy and
    determine optimal scenario distribution.
    """
    
    async def execute(self, policy: Policy, traces: int, planner: Planner) -> Plan:
        """
        Execute the planning phase.
        
        Args:
            policy: The policy to analyze
            traces: Target number of traces
            planner: Planner component to use
            
        Returns:
            Plan with categories and trace distribution
        """
        return await planner.plan(policy.text, traces)


class ScenarioPhase:
    """
    Scenario generation phase - creates scenarios for each category.
    
    Runs in parallel across categories for efficiency.
    """
    
    async def execute(
        self,
        policy: Policy,
        plan: Plan,
        generator: ScenarioGenerator,
        semaphore: Semaphore,
    ) -> list[Scenario]:
        """
        Execute scenario generation.
        
        Args:
            policy: The policy text
            plan: Plan with categories
            generator: ScenarioGenerator component
            semaphore: Semaphore for rate limiting
            
        Returns:
            List of all generated scenarios
        """
        async def limited_generate(category):
            async with semaphore:
                return await generator.generate(policy.text, category.count, category=category)
        
        tasks = [limited_generate(cat) for cat in plan.categories]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [scenario for batch in results for scenario in batch]


class ResponsePhase:
    """
    Response generation phase - creates responses for each scenario.
    
    Runs fully parallel with semaphore control.
    """
    
    async def execute(
        self,
        policy: Policy,
        scenarios: list[Scenario],
        generator: ResponseGenerator,
        semaphore: Semaphore,
    ) -> list[Trace]:
        """
        Execute response generation.
        
        Args:
            policy: The policy text
            scenarios: List of scenarios to respond to
            generator: ResponseGenerator component
            semaphore: Semaphore for rate limiting
            
        Returns:
            List of traces with generated responses
        """
        async def limited_generate(scenario):
            async with semaphore:
                return await generator._generate_single(policy.text, scenario)
        
        tasks = [limited_generate(s) for s in scenarios]
        return await asyncio.gather(*tasks)


class GradingPhase:
    """
    Grading and refinement phase - evaluates and improves responses.
    
    Includes the refinement loop for failed traces.
    """
    
    async def execute(
        self,
        policy: Policy,
        traces: list[Trace],
        grader: Grader,
        refiner: Refiner,
        max_iterations: int,
        semaphore: Semaphore,
    ) -> tuple[list[Trace], float]:
        """
        Execute grading and refinement.
        
        Args:
            policy: The policy text
            traces: List of traces to grade
            grader: Grader component
            refiner: Refiner component
            max_iterations: Maximum refinement iterations
            semaphore: Semaphore for rate limiting
            
        Returns:
            Tuple of (graded traces, pass rate percentage)
        """
        async def limited_grade(trace):
            async with semaphore:
                return await grader.grade(trace, policy.text)
        
        async def limited_refine(trace, grade):
            async with semaphore:
                return await refiner.refine(trace, grade, policy.text)
        
        # Initial grading
        grade_tasks = [limited_grade(t) for t in traces]
        grades = await asyncio.gather(*grade_tasks)
        
        # Attach grades
        final_traces = list(traces)
        for trace, grade in zip(final_traces, grades):
            trace.grade = grade
        
        # Refinement loop
        for iteration in range(1, max_iterations):
            failed_indices = [i for i, t in enumerate(final_traces) if not t.grade.passed]
            
            if not failed_indices:
                break
            
            # Refine failed traces
            refine_tasks = [
                limited_refine(final_traces[i], final_traces[i].grade)
                for i in failed_indices
            ]
            refined_traces = await asyncio.gather(*refine_tasks)
            
            # Preserve original scenarios and update traces
            for idx, refined in zip(failed_indices, refined_traces):
                refined.scenario = final_traces[idx].scenario
                final_traces[idx] = refined
            
            # Re-grade refined traces
            regrade_tasks = [limited_grade(final_traces[i]) for i in failed_indices]
            new_grades = await asyncio.gather(*regrade_tasks)
            
            for idx, grade in zip(failed_indices, new_grades):
                final_traces[idx].grade = grade
        
        # Calculate pass rate
        passed_count = sum(1 for t in final_traces if t.grade and t.grade.passed)
        pass_rate = (passed_count / len(final_traces) * 100) if final_traces else 0
        
        return final_traces, pass_rate


__all__ = ["PlanPhase", "ScenarioPhase", "ResponsePhase", "GradingPhase"]

