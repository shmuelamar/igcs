from pathlib import Path

# Main Dirs
ROOT_DIR = Path(__file__).parents[2].resolve()
SRC_DATASETS_DIR = ROOT_DIR / "source_datasets"
RAW_TRAIN_DATASETS_DIR = ROOT_DIR / "raw_train_datasets"
PREDICTIONS_DIR = ROOT_DIR / "models_predictions"
EVAL_RESULTS_DIR = ROOT_DIR / "models_evaluations"
VISUALIZED_PREDICTIONS = ROOT_DIR / "visualized_predictions"
IGCS_DATA_DIR = ROOT_DIR / "igcs-dataset"
CACHE_DIR = ROOT_DIR / "cache"

# RI Source Filenames
PG19_FNAME = RAW_TRAIN_DATASETS_DIR / "pg19-books-upto-3500-words.jsonl.gz"
ENRON_FNAME = RAW_TRAIN_DATASETS_DIR / "enron_emails.jsonl.gz"
MNEWS_FNAME = RAW_TRAIN_DATASETS_DIR / "MultiNews" / "train.jsonl.gz"
TRIPADVISOR_FNAME = RAW_TRAIN_DATASETS_DIR / "trip_advisor.jsonl.gz"
GITHUB_FNAME = RAW_TRAIN_DATASETS_DIR / "github_code_clustered.jsonl.gz"
WIKI_FNAME = RAW_TRAIN_DATASETS_DIR / "wiki_pages_100k.jsonl.gz"
PUBMED_FNAME = RAW_TRAIN_DATASETS_DIR / "pubmed_abstracts_100k.jsonl.gz"

# RI Dataset Paths
RI_DATASET_DIR = SRC_DATASETS_DIR / "ReverseInstructions"
SAMPLED_RI_RAW_DATA_FNAME = (
    RAW_TRAIN_DATASETS_DIR / "sampled_ri_dataset_for_annotation_n500.jsonl.gz"
)
ANNOTATED_RI_DIR = RAW_TRAIN_DATASETS_DIR / "annotated_ri"
ANNOTATED_RI_FNAME = ANNOTATED_RI_DIR / "annotated_ri_samples.jsonl.gz"

# Dataset Dir
TRAIN_DIR = IGCS_DATA_DIR / "train"
DEV_DIR = IGCS_DATA_DIR / "dev"
TEST_DIR = IGCS_DATA_DIR / "test"
PROMPTS_DIR = IGCS_DATA_DIR / "prompts"
