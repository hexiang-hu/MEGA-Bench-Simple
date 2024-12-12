import os
import logging
from datasets import load_dataset
import pandas as pd
from evaluator import MegaEvaluator, WeightedAverageAccumulator
from xlm.evals.multimodal.utils import extract_multiple_choice
from urllib.request import urlretrieve
from zipfile import ZipFile

import nltk
from nltk.corpus import wordnet
try:
    nltk.data.find("tokenizers/punkt")
    print("The 'punkt' tokenizer is already available.")
except LookupError:
    print("Downloading 'punkt' tokenizer...")
    nltk.download("punkt")

try:
    wordnet.synsets("test")
    print("The 'wordnet' corpus is already available.")
except LookupError:
    print("Downloading 'wordnet' corpus...")
    nltk.download("wordnet")


logger = logging.getLogger(__name__)

# Configuration
DATA_ROOT_PATH = "data/raw/"
SPLITS = ["core", "core_single_image"]

# Download and extract data
def download_and_extract_data():
    url = "https://huggingface.co/datasets/TIGER-Lab/MEGA-Bench/resolve/main/data.zip?download=true"
    zip_path, _ = urlretrieve(url, "data.zip")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("megabench")
    os.rename("megabench/data", DATA_ROOT_PATH)

# Load dataset
def load_datasets():
    return {
        split: load_dataset("TIGER-Lab/MEGA-Bench", split)
        for split in SPLITS
    }


def _load_media_content(media_path):
    # normalize media path
    media_path = media_path.replace("./", DATA_ROOT_PATH)
    if is_video_file(media_path):
        images = video_utils.read_video(media_path, NUM_VIDEO_FRAMES)
    else:
        images = []
        with open(os.path.join(media_path), "rb") as fd:
            images.append(fd.read())

    return [{"type": "image", "content": image_bytes} for image_bytes in images]


def _process_text_and_media(text, media_paths):
    content = []
    chunks = re.split(r"(<image>|<video>)", text)

    if isinstance(media_paths, str):
        media_paths = eval(media_paths)

    placeholder_count = sum(1 for chunk in chunks if chunk in ["<image>", "<video>"])
    if placeholder_count != len(media_paths):
        raise ValueError(
            f"Mismatching # placeholders ({placeholder_count}) and # media paths ({len(media_paths)})"
        )

    media_index = 0
    for chunk in chunks:
        if chunk in ["<image>", "<video>"]:
            media_content = _load_media_content(media_paths[media_index])
            if len(media_content) == 1:  # image
                content.extend(media_content)
            else:  # video
                content.extend(media_content)
            media_index += 1
        elif chunk.strip():
            content.append({"type": "text", "content": chunk.strip()})

    return content


# Simplified MegaBenchDatasetReader
class MegaBenchDatasetReader:
    def __init__(self, split):
        self.split = split
        self.ds = load_dataset("TIGER-Lab/MEGA-Bench", self.split)

    def get_data(self):
        return self.ds['test']

    def get_conversation(self, datapoint):
        # Simplified conversation structure
        human_data_chunks = _process_text_and_media(
            datapoint["task_description"], datapoint["global_media"]
        )
        human_data_chunks.extend(
            _process_text_and_media(datapoint["example_text"], datapoint["example_media"])
        )
        human_data_chunks.extend(
            _process_text_and_media(datapoint["query_text"], datapoint["query_media"])
        )

        return [{"role": "Human", "chunks": human_data_chunks}, {"role": "Assistant", "chunks": [{"type": "text", "content": ""}]}]
    
    def get_target(self, datapoint):
        return datapoint["answer"]

    def get_extra_fields(self, datapoint):
        return {
            "question_types": datapoint["input_format"],
            "answer_types": datapoint["output_format"],
            "task_names": datapoint["task_name"],
            "task_trees": datapoint["taxonomy_tree_path"],
            "subsets": datapoint["application"],
            "metric_infos": datapoint["metric_info"],
            "eval_contexts": datapoint["eval_context"],
        }

# Simplified MegaBenchEvalMetrics
class MegaBenchEvalMetrics:
    def __init__(self):
        self.evaluator = MegaEvaluator()
        self.accumulators = {"overall": WeightedAverageAccumulator()}

    def eval_prediction(self, batch):
        predictions = [""] * len(batch["targets"])  # Empty predictions for metric computation
        for i, (prediction, target, metric_info, eval_context) in enumerate(zip(predictions, batch["targets"], batch["metric_infos"], batch["eval_contexts"])):
            metrics, score_function_table = self.evaluator.load_task_metrics(metric_info)
            response_obj, _ = self.evaluator.parse_response(prediction, target, score_function_table)
            field_scores, _ = self.evaluator.score_fields(response_obj, target, metrics, eval_context)
            aggregated_score = self.evaluator.aggregate_scores(field_scores, score_function_table["aggregation"])

            self.accumulators["overall"].add(aggregated_score, 1)
            logger.info(f"{'✅' if aggregated_score > 0.5 else '❌'} | Pred: {prediction} | Target: {target}")

    def get(self):
        acc = self.accumulators["overall"].get()
        num_samples = int(self.accumulators["overall"].weight)
        return [{"subset": "overall", "acc": acc, "num_samples": num_samples}]


def dummy_model(batch):
	batch["predictions"] = []
	for datum in batch["targets"]:
		batch["predictions"].append(datum)

	return batch


# Main execution
if __name__ == "__main__":
    download_and_extract_data()
    datasets = load_datasets()
    
    for split, ds in datasets.items():
        reader = MegaBenchDatasetReader(split)
        data = reader.get_data()

        # Prepare batch for evaluation
        batch = {
            "targets": [reader.get_target(d) for d in data],
            "metric_infos": [reader.get_extra_fields(d)["metric_infos"] for d in data],
            "eval_contexts": [reader.get_extra_fields(d)["eval_contexts"] for d in data]
        }
        
        batch = dummy_model(batch)

        metrics = MegaBenchEvalMetrics()
        metrics.eval_prediction(batch)
        
        results = metrics.get()
        for result in results:
            logger.info(f"{result['subset']}: {result['acc']:.4f} ({result['num_samples']} samples)")