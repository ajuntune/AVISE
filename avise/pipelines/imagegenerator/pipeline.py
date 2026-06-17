"""Base class for all Image Generator Security Evaluation Tests.

All image generator SETs inherit from BaseImageGenSETPipeline and implement
the same 4-phase pipeline contract used by LM SETs:

    initialize() -> execute() -> evaluate() -> report()

The pipeline reuses EvaluationResult and ReportData from the language-model
pipeline schema so that the existing reporters (JSON, HTML, Markdown) and the
AI summariser work unchanged across both model types.

Data flow:
    initialize() ---> List[ImageGenSETCase]
        ---> execute() ---> ImageGenOutputData
        ---> evaluate() ---> List[EvaluationResult]
        ---> report() ---> ReportData
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from math import sqrt
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from scipy.special import erfinv

from .schema import ImageGenSETCase, ImageGenOutputData
from ..languagemodel.schema import EvaluationResult, ReportData
from ...utils.report_format import ReportFormat

if TYPE_CHECKING:
    # Only imported for type-checking; never at runtime to avoid circular imports.
    from ...connectors.imagegenerator.base import BaseImageGenConnector

logger = logging.getLogger(__name__)


class BaseImageGenSETPipeline(ABC):
    """Abstract base pipeline for Image Generator Security Evaluation Tests.

    Mirrors BaseSETPipeline (languagemodel) so that the engine, CLI, and
    registry work the same way for image-generator SETs as for LM SETs.

    Phase 1 - initialize(set_config_path) -> List[ImageGenSETCase]
    Phase 2 - execute(connector, sets)    -> ImageGenOutputData
    Phase 3 - evaluate(execution_data)   -> List[EvaluationResult]
    Phase 4 - report(results, ...)       -> ReportData
    """

    name: str = ""
    description: str = ""
    SUPPORTED_FORMATS = [ReportFormat.JSON, ReportFormat.HTML, ReportFormat.MARKDOWN]

    def __init__(self) -> None:
        self.set_cases: List[ImageGenSETCase] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.connector_config_path: Optional[str] = None
        self.set_config_path: Optional[str] = None
        self.target_model_name: Optional[str] = None
        # evaluation_model_name is accepted but unused in the first SET.
        # Future SETs that use a vision model will set this.
        self.evaluation_model_name: Optional[str] = None
        self.evaluation_model = None

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement all four methods
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, set_config_path: str) -> List[ImageGenSETCase]:
        """Load and return SET cases from a configuration file.

        Args:
            set_config_path: Path to the SET configuration file.

        Returns:
            List[ImageGenSETCase]: SET cases for this run.
        """
        pass

    @abstractmethod
    def execute(
        self,
        connector: BaseImageGenConnector,
        sets: List[ImageGenSETCase],
    ) -> ImageGenOutputData:
        """Run each SET case against the target image generator.

        Args:
            connector: An initialised image-generator connector.
            sets: SET cases from initialize().

        Returns:
            ImageGenOutputData: All execution outputs + wall-clock duration.
        """
        pass

    @abstractmethod
    def evaluate(
        self, execution_data: ImageGenOutputData
    ) -> List[EvaluationResult]:
        """Evaluate execution outputs.

        Args:
            execution_data: ImageGenOutputData from execute().

        Returns:
            List[EvaluationResult]: One result per execution output.
            Status must be "passed", "failed", or "error".
        """
        pass

    @abstractmethod
    def report(
        self,
        results: List[EvaluationResult],
        output_path: str,
        report_format: ReportFormat = ReportFormat.JSON,
        generate_ai_summary: bool = True,
    ) -> ReportData:
        """Generate and write the final report.

        Args:
            results: List[EvaluationResult] from evaluate().
            output_path: Destination file path.
            report_format: Desired output format.
            generate_ai_summary: Whether to request an AI-generated summary.

        Returns:
            ReportData: The final report object.
        """
        pass

    # ------------------------------------------------------------------
    # run() – orchestration (called by ExecutionEngine)
    # ------------------------------------------------------------------

    def run(
        self,
        connector: BaseImageGenConnector,
        set_config_path: str,
        output_path: str,
        report_format: ReportFormat = ReportFormat.JSON,
        connector_config_path: Optional[str] = None,
        generate_ai_summary: bool = True,
        runs: int = 1,
    ) -> ReportData:
        """Orchestrate the 4-phase pipeline.

        Called by ExecutionEngine.run_test(). Signature matches
        BaseSETPipeline.run() so the engine needs no special-casing.

        Args:
            connector: Initialised image-generator connector.
            set_config_path: Path to the SET configuration.
            output_path: Where the output report is written.
            report_format: Desired output format.
            connector_config_path: Path to the connector config (for report metadata).
            generate_ai_summary: Whether to generate an AI-powered summary.
            runs: How many times to repeat the SET.

        Returns:
            ReportData: The final report.
        """
        self.connector_config_path = connector_config_path
        self.set_config_path = set_config_path
        self.target_model_name = getattr(connector, "model", connector.name)

        try:
            sets = self.initialize(set_config_path)

            results: List[EvaluationResult] = []
            for run in range(runs):
                logger.info(f"Starting SET run {run + 1}/{runs}.")
                execution_data = self.execute(connector, sets)
                results += self.evaluate(execution_data)
                logger.info(f"SET run {run + 1}/{runs} finished.")

            report_data = self.report(
                results, output_path, report_format, generate_ai_summary
            )
            return report_data

        finally:
            # Clean up any vision evaluation model if one was loaded
            if self.evaluation_model is not None:
                if hasattr(self.evaluation_model, "del_model"):
                    self.evaluation_model.del_model()
                self.evaluation_model = None

    # ------------------------------------------------------------------
    # Shared helpers (identical to BaseSETPipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_passrates(results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics (pass%, fail%, error%) from results."""
        total = len(results)
        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        errors = total - passed - failed

        pass_rate = round(passed / total * 100, 1) if total > 0 else 0
        fail_rate = round(failed / total * 100, 1) if total > 0 else 0

        ci = BaseImageGenSETPipeline._calculate_confidence_interval(passed, failed)
        return {
            "total_set_cases": total,
            "passed": passed,
            "failed": failed,
            "error": errors,
            "pass_rate": pass_rate,
            "fail_rate": fail_rate,
            "ci_lower_bound": ci[1],
            "ci_upper_bound": ci[2],
        }

    @staticmethod
    def calculate_subcategory_runs(
        results: List[EvaluationResult],
        subcategory_field: str = "harm_category",
    ) -> Dict[str, int]:
        """Count runs per harm-category (subcategory)."""
        subcategory_runs: Dict[str, int] = {}
        for result in results:
            cat = result.metadata.get(subcategory_field, "Unknown")
            subcategory_runs[cat] = subcategory_runs.get(cat, 0) + 1
        return subcategory_runs

    @staticmethod
    def _calculate_confidence_interval(
        passed: int, failed: int, confidence_level: float = 0.95
    ):
        """Wilson score confidence interval for pass rate."""
        n = passed + failed
        if n == 0:
            return (0, 0, 0)
        p = passed / n
        z = 1.96 if confidence_level == 0.95 else sqrt(2) * erfinv(confidence_level)
        denominator = 1 + (z**2 / n)
        center = (p + (z**2 / (2 * n))) / denominator
        margin = (z / denominator) * sqrt(
            (p * (1 - p) / n) + (z**2 / (4 * n**2))
        )
        return (p, max(0, center - margin), min(1, center + margin))
