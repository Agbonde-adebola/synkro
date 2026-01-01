"""Pipeline runner that orchestrates all phases."""

import asyncio
from datetime import datetime

from synkro.core.policy import Policy
from synkro.core.dataset import Dataset
from synkro.factory import ComponentFactory
from synkro.reporting import ProgressReporter
from synkro.pipeline.phases import PlanPhase, ScenarioPhase, ResponsePhase, GradingPhase


class GenerationPipeline:
    """
    Orchestrates the full generation pipeline using decomposed phases.
    
    This class coordinates the execution of all phases and reports
    progress through the injected reporter.
    
    Examples:
        >>> pipeline = GenerationPipeline(factory, reporter, workers=10)
        >>> dataset = await pipeline.run(policy, traces=50)
    """
    
    def __init__(
        self,
        factory: ComponentFactory,
        reporter: ProgressReporter,
        workers: int,
        max_iterations: int = 1,
        skip_grading: bool = False,
    ):
        """
        Initialize the pipeline.
        
        Args:
            factory: ComponentFactory for creating pipeline components
            reporter: ProgressReporter for reporting progress
            workers: Number of concurrent workers (API calls)
            max_iterations: Maximum refinement iterations
            skip_grading: Whether to skip the grading phase
        """
        self.factory = factory
        self.reporter = reporter
        self.workers = workers
        self.max_iterations = max_iterations
        self.skip_grading = skip_grading
        
        # Phases
        self.plan_phase = PlanPhase()
        self.scenario_phase = ScenarioPhase()
        self.response_phase = ResponsePhase()
        self.grading_phase = GradingPhase()
    
    async def run(self, policy: Policy, traces: int, model: str, dataset_type: str) -> Dataset:
        """
        Run the full generation pipeline.
        
        Args:
            policy: The policy to generate from
            traces: Target number of traces
            model: Model name (for reporting)
            dataset_type: Dataset type (for reporting)
            
        Returns:
            Dataset with generated traces
        """
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(self.workers)
        
        # Create components via factory
        planner = self.factory.create_planner()
        scenario_gen = self.factory.create_scenario_generator()
        response_gen = self.factory.create_response_generator()
        grader = self.factory.create_grader()
        refiner = self.factory.create_refiner()
        
        # Report start
        self.reporter.on_start(traces, model, dataset_type)
        
        # Phase 1: Planning
        plan = await self.plan_phase.execute(policy, traces, planner)
        self.reporter.on_plan_complete(plan)
        
        # Phase 2: Scenario generation
        scenarios = await self.scenario_phase.execute(policy, plan, scenario_gen, semaphore)
        self.reporter.on_scenarios_complete(scenarios)
        
        # Phase 3: Response generation
        all_traces = await self.response_phase.execute(policy, scenarios, response_gen, semaphore)
        self.reporter.on_responses_complete(list(all_traces))
        
        # Phase 4: Grading (optional)
        pass_rate: float | None = None
        if self.skip_grading:
            final_traces = list(all_traces)
            self.reporter.on_grading_skipped()
        else:
            final_traces, pass_rate = await self._run_grading_with_reporting(
                policy, list(all_traces), grader, refiner, semaphore
            )
            self.reporter.on_grading_complete(final_traces, pass_rate)
        
        # Report completion
        elapsed = (datetime.now() - start_time).total_seconds()
        self.reporter.on_complete(len(final_traces), elapsed, pass_rate)
        
        return Dataset(traces=final_traces)
    
    async def _run_grading_with_reporting(
        self,
        policy: Policy,
        traces: list,
        grader,
        refiner,
        semaphore,
    ) -> tuple[list, float]:
        """Run grading phase with refinement iteration reporting."""
        
        async def limited_grade(trace):
            async with semaphore:
                return await grader.grade(trace, policy.text)
        
        async def limited_refine(trace, grade):
            async with semaphore:
                return await refiner.refine(trace, grade, policy.text)
        
        # Initial grading
        grade_tasks = [limited_grade(t) for t in traces]
        grades = await asyncio.gather(*grade_tasks)
        
        final_traces = list(traces)
        for trace, grade in zip(final_traces, grades):
            trace.grade = grade
        
        # Refinement loop with reporting
        for iteration in range(1, self.max_iterations):
            failed_indices = [i for i, t in enumerate(final_traces) if not t.grade.passed]
            
            if not failed_indices:
                break
            
            self.reporter.on_refinement_start(iteration + 1, len(failed_indices))
            
            # Refine
            refine_tasks = [
                limited_refine(final_traces[i], final_traces[i].grade)
                for i in failed_indices
            ]
            refined_traces = await asyncio.gather(*refine_tasks)
            
            for idx, refined in zip(failed_indices, refined_traces):
                refined.scenario = final_traces[idx].scenario
                final_traces[idx] = refined
            
            # Re-grade
            regrade_tasks = [limited_grade(final_traces[i]) for i in failed_indices]
            new_grades = await asyncio.gather(*regrade_tasks)
            
            for idx, grade in zip(failed_indices, new_grades):
                final_traces[idx].grade = grade
        
        passed_count = sum(1 for t in final_traces if t.grade and t.grade.passed)
        pass_rate = (passed_count / len(final_traces) * 100) if final_traces else 0
        
        return final_traces, pass_rate


__all__ = ["GenerationPipeline"]

