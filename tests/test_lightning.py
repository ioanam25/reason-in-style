import pytest
import pytorch_lightning as pl
import torch
from unittest.mock import MagicMock, patch

from src.module import TraceDataset, collate_trace_pairs, TraceCollator

@pytest.fixture
def mock_data():
    return [
        {'question_id': '1', 'question': 'q1', 'trace': 'trace 1a', 'answer': '1'},
        {'question_id': '1', 'question': 'q1', 'trace': 'trace 1b', 'answer': '1'},
        {'question_id': '2', 'question': 'q2', 'trace': 'trace 2a', 'answer': '2'},
        {'question_id': '2', 'question': 'q2', 'trace': 'trace 2b', 'answer': '2'},
    ]

@patch('pytorch_lightning.Trainer.fit')
def test_fast_dev_run(mock_fit, mock_data):
    """Test 4.2a: Fast Dev Run MOCKED."""
    module = MagicMock()
    datamodule = MagicMock()
    
    trainer = pl.Trainer(
        fast_dev_run=True,
        enable_checkpointing=False,
        logger=False
    )
    
    trainer.fit(module, datamodule)
    mock_fit.assert_called_once_with(module, datamodule)

@patch('pytorch_lightning.Trainer.fit')
def test_overfit_batch(mock_fit, mock_data):
    """Test 4.2b: Overfit Batch MOCKED."""
    module = MagicMock()
    datamodule = MagicMock()
    
    trainer = pl.Trainer(
        max_epochs=5,
        overfit_batches=1,
        enable_checkpointing=False,
        logger=False
    )
    
    trainer.fit(module, datamodule)
    mock_fit.assert_called_once_with(module, datamodule)


def test_trace_dataset_builds_all_unique_positive_pairs():
    data = [
        {'question_id': '1', 'question': 'q1', 'trace': 'trace 1a', 'answer': '1'},
        {'question_id': '1', 'question': 'q1', 'trace': 'trace 1b', 'answer': '1'},
        {'question_id': '1', 'question': 'q1', 'trace': 'trace 1c', 'answer': '1'},
        {'question_id': '1', 'question': 'q1', 'trace': 'trace 1d', 'answer': '1'},
    ]

    dataset = TraceDataset(data)

    assert len(dataset) == 6
    assert dataset[0][0]['question_id'] == '1'
    assert dataset[0][1]['question_id'] == '1'
    observed_pairs = {
        tuple(sorted((left['trace'], right['trace'])))
        for left, right in dataset.paired_data
    }
    assert observed_pairs == {
        ('trace 1a', 'trace 1b'),
        ('trace 1a', 'trace 1c'),
        ('trace 1a', 'trace 1d'),
        ('trace 1b', 'trace 1c'),
        ('trace 1b', 'trace 1d'),
        ('trace 1c', 'trace 1d'),
    }


def test_collate_trace_pairs_preserves_pair_adjacency():
    batch = [
        (
            {'question_id': '1', 'question': 'q1', 'trace': 'trace 1a', 'answer': '1'},
            {'question_id': '1', 'question': 'q1', 'trace': 'trace 1c', 'answer': '1'},
        ),
        (
            {'question_id': '2', 'question': 'q2', 'trace': 'trace 2a', 'answer': '2'},
            {'question_id': '2', 'question': 'q2', 'trace': 'trace 2b', 'answer': '2'},
        ),
    ]

    # Legacy collate returns raw strings
    traces = collate_trace_pairs(batch)
    assert traces == ['trace 1a', 'trace 1c', 'trace 2a', 'trace 2b']

    # TraceCollator returns pre-tokenized dict
    collator = TraceCollator(encoder_tokenizer=None, decoder_tokenizer=None, max_len=32, vocab_size=1000)
    result = collator(batch)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 4
    assert result["attention_mask"].shape[0] == 4
    assert result["input_ids"].shape[1] == 32  # padded to max_len


def test_trace_collator_prepends_decoder_start_token():
    collator = TraceCollator(
        encoder_tokenizer=None, decoder_tokenizer=None, max_len=32, vocab_size=1000, decoder_start_token_id=42,
    )
    batch = [
        (
            {"question_id": "1", "question": "q1", "trace": "ab", "answer": "1"},
            {"question_id": "1", "question": "q1", "trace": "cd", "answer": "1"},
        ),
    ]
    result = collator(batch)
    assert result["input_ids"].shape == (2, 32)
    assert (result["input_ids"][:, 0] == 42).all()
    assert (result["attention_mask"][:, 0] == 1).all()
