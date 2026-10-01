from argparse import Namespace

from datasets import Dataset

from main import resolve_datasets


def make_args(**overrides):
    values = {
        "data_path": None,
        "val_data_path": None,
        "validation_fraction": 0.1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_datasets_uses_smoke_test_data_without_disk_path():
    train_records, validation_records, data_mode = resolve_datasets(make_args())

    assert data_mode == "smoke-test"
    assert train_records == validation_records
    assert len(train_records) == 4
    assert {record["question_id"] for record in train_records} == {"1", "2"}


def test_resolve_datasets_splits_single_disk_dataset_by_question_id(tmp_path):
    dataset = Dataset.from_list(
        [
            {"question_id": "1", "question": "q1", "trace": "trace 1a", "answer": "1"},
            {"question_id": "1", "question": "q1", "trace": "trace 1b", "answer": "1"},
            {"question_id": "2", "question": "q2", "trace": "trace 2a", "answer": "2"},
            {"question_id": "2", "question": "q2", "trace": "trace 2b", "answer": "2"},
            {"question_id": "3", "question": "q3", "trace": "trace 3a", "answer": "3"},
            {"question_id": "3", "question": "q3", "trace": "trace 3b", "answer": "3"},
        ]
    )
    data_path = tmp_path / "train_dataset"
    dataset.save_to_disk(str(data_path))

    train_records, validation_records, data_mode = resolve_datasets(
        make_args(data_path=str(data_path), validation_fraction=0.34)
    )

    assert data_mode == "disk"
    assert {record["question_id"] for record in train_records} == {"1", "2"}
    assert {record["question_id"] for record in validation_records} == {"3"}
    assert len(train_records) == 4
    assert len(validation_records) == 2