import ast
import dataclass
from typing import Any, Dict, Tuple

from metrics.aggregation_type import AggregationType
from metrics.metric_type import MetricType
from metrics.parsing.common.utils import evaluate_as_string
from metrics.response_parse_type import ResponseParseType


@dataclass
class WeightedAverageAccumulator:
    sum: float = 0.0
    weight: float = 0.0

    def add(self, metrics, weight):
        self.sum += metrics
        self.weight += weight

    def get(self):
        if self.weight == 0.0:
            return jnp.array(0.0)
        else:
            return self.sum / self.weight


class MegaEvaluator:
    @staticmethod
    def load_task_metrics(metric_info):
        """Load and initialize metrics from metric info dictionary."""
        metrics = {}
        score_function_table = (
            ast.literal_eval(metric_info) if isinstance(metric_info, str) else metric_info
        )

        for field, scoring_function_name in score_function_table["field_score_function"].items():
            metric = MetricType.from_string(scoring_function_name)
            metrics[field] = metric

        # Handle global auxiliary metrics
        global_aux_fields = score_function_table.get("global_aux_metrics", {})
        for field, scoring_function_name in global_aux_fields.items():
            metric = MetricType.from_string(scoring_function_name)
            metrics[field] = metric

        return metrics, score_function_table

    def parse_response(
        self,
        prediction: str,
        ground_truth: Dict[str, Any],
        score_function_table: Dict[str, Any],
        query_text: str = "",
    ) -> Tuple[Dict, bool]:
        """
        Parse the prediction response based on the specified parser type.

        Args:
            prediction: Model's prediction string
            parser: Parser type to use
            ground_truth: Dictionary containing correct answers for each field
            query_text: Optional query text for certain parsers

        Returns:
            Tuple of (parsed response object, parsing success boolean)
        """
        parser = ResponseParseType.from_string(score_function_table["response_parse_function"])
        if parser.is_single_field_parser():
            answer_fields = [field for field in ground_truth.keys() if not field.startswith("##")]
            assert (
                len(answer_fields) == 1
            ), "Single field parser must be used with single field answers"
            answer_key = answer_fields[0]

            is_single_line_ans = "\n" not in ground_truth[answer_key]
            response_obj = parser.parse(
                prediction,
                answer_key,
                global_description="",
                query_question=query_text,
                is_single_line_ans=is_single_line_ans,
            )
            assert isinstance(
                response_obj[answer_key], str
            ), "Single-field parsing results must be string"
            return response_obj, True
        else:
            response_obj = parser.parse(prediction)
            if parser == ResponseParseType.JSON and (
                not isinstance(response_obj, dict) or not response_obj
            ):
                return {field: prediction for field in ground_truth}, False
            return response_obj, True

    def score_fields(
        self,
        response_obj: Dict,
        ground_truth: Dict[str, Any],
        metrics: Dict,
        eval_context: Dict,
    ) -> Tuple[Dict, Dict]:
        """
        Score each field in the response using the specified metrics.

        Args:
            response_obj: Parsed response object
            ground_truth: Dictionary containing correct answers
            metrics: Dictionary of metrics to use for each field
            eval_context: evaluation context

        Returns:
            Tuple of (field scores dictionary, field info dictionary)
        """
        field_scores = {}
        field_info = {}

        for field, correct_value in ground_truth.items():
            metric = metrics[field]
            correct_value = evaluate_as_string(correct_value)

            if metric == MetricType.UNSUPPORTED:
                field_scores[field] = 0
            elif metric == MetricType.SYMBOLIC_PLANNING_TEST:
                field_scores[field] = metric.match(
                    response_obj.get(field),
                    eval_context,
                )
            elif metric == MetricType.PROGRAM_JUDGE:
                field_scores[field] = metric.match(
                    response_obj.get(field),
                    eval_context,
                )
            elif metric == MetricType.CONSTRAINED_GENERATION:
                score, eval_info = metric.match(response_obj, eval_context)
                field_scores[field] = score
                field_info[field] = eval_info
            elif metric == MetricType.XML_NORM_POINT_IN_BBOX:
                score, eval_info = metric.match(response_obj.get(field), eval_context)
                field_scores[field] = score
                field_info[field] = eval_info
            else:
                field_scores[field] = metric.match(response_obj.get(field), correct_value)

        return field_scores, field_info

    def aggregate_scores(self, field_scores: Dict[str, float], aggregation_config: Dict) -> float:
        aggregator = AggregationType.from_string(aggregation_config["function"])
        return aggregator.aggregate(field_scores, aggregation_config["field_weights"])
