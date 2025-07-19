from .aspectnews import AspectNewsDataset
from .debatesum import DebateSumDataset
from .openasp import OpenAspDataset
from .reverse_instructions import ReverseInstructionsDataset
from .scifact import SciFactDataset
from .spark import SparkEvidenceDataset, SparkSaliencyDataset

DATASETS = (
    AspectNewsDataset,
    OpenAspDataset,
    SciFactDataset,
    DebateSumDataset,
    SparkSaliencyDataset,
    SparkEvidenceDataset,
    ReverseInstructionsDataset,
)

NAME2DATASET = {ds.name: ds for ds in DATASETS}
