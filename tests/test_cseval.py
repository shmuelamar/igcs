import json

import pandas as pd

from igcs import cseval
from igcs.entities import Doc, SelectionGroup, SelectionSample
from igcs.grounding import ground_selection
from igcs.utils import ioutils

IDS = (
    "enron.lay-k/sent/46..inst-0",
    "enron.lay-k/sent/46..inst-1",
    "tripadvisor.796339.inst-3",
    "tripadvisor.796339.inst-4",
    "github.kikov79/scalr/app/src/Scalr/Model/Entity.inst-4",
)

ID0 = IDS[0]
ID1 = IDS[1]

SELECTIONS = (
    json.dumps([]),
    json.dumps(["Hello world", "one two 3"]),
    json.dumps(["long " * 15]),
    json.dumps(["abc"] * 10),
    ["abd", "def", "ghi"],
)
INVALID_SELECTIONS = ("][[]", "[None", "not formatted as relevant content")

DOCS1 = [
    Doc(id=0, filename="1.txt", text="Hello world.\n" + "long " * 15 + "done.\n"),
    Doc(
        id=1,
        filename="2.txt",
        text="abc very selection long 1234.\none more line.\nand even one more.",
    ),
]
DOCS2 = [
    Doc(id=0, filename="3.txt", text="nothing here."),
    Doc(
        id=1,
        filename="4.py",
        text="print('hello world')\n# abd\ndef ghi(): return True\ns = 'one two 3'",
    ),
]

REF_DATA = [
    SelectionSample(
        id=IDS[0],
        source_dataset="TaggedDataset",
        selections=[
            SelectionGroup(
                id="1",
                selections=[
                    ground_selection("Hello world", DOCS1),
                    ground_selection("long " * 13, DOCS1),
                ],
            ),
            SelectionGroup(
                id="2",
                selections=[
                    ground_selection("long " * 15, DOCS1),
                ],
            ),
        ],
        docs=DOCS1,
        instruction_context={},
    ),
    SelectionSample(
        id=IDS[1],
        source_dataset="TaggedDataset",
        selections=[
            SelectionGroup(
                selections=[
                    ground_selection("abd", DOCS2),
                    ground_selection("def", DOCS2),
                    ground_selection("ghi", DOCS2),
                    ground_selection("print(", DOCS2),
                ]
            )
        ],
        docs=DOCS2,
        instruction_context={},
    ),
]


PRED_DATA = [
    {"sample_id": ID0, "selection": json.dumps(["Hello world", "one two 3"])},
    {"sample_id": ID1, "selection": "[ Invalid"},
]


def test_cseval_main(tmpdir):
    ref_file = tmpdir.join("ref.jsonl")
    pred_file = tmpdir.join("pred.jsonl")
    outfile = tmpdir.join("out.jsonl")
    ioutils.jsonl_dump([x.model_dump(mode="json") for x in REF_DATA], ref_file)
    ioutils.jsonl_dump(PRED_DATA, pred_file)

    cseval.main(["-r", str(ref_file), "-p", str(pred_file), "-o", str(outfile)])

    df = pd.read_json(outfile, lines=True)
    assert len(df) == 2
    assert df["f1"].round(3).tolist() == [0.222, 0.0]
