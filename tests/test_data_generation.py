import pytest
import os
import torch
from src.data_generation import GenerationPipeline

def test_mock_generator():
    """Test 1.3a: Implement a dummy Hugging Face pipeline that yields randomized string completions."""
    pipeline = GenerationPipeline(mock=True)
    questions = ["What is 2+2?", "Solve 3x=9"]
    ids = ["q1", "q2"]
    K = 4
    
    dataset = pipeline.generate_traces(questions, ids, k=K)
    
    assert len(dataset) == len(questions) * K
    assert "question_id" in dataset.column_names
    assert "question" in dataset.column_names
    assert "trace" in dataset.column_names
    assert "answer" in dataset.column_names
    
    # Save test to arrow
    dataset.save_to_disk("mock_dataset_arrow")
    assert os.path.exists("mock_dataset_arrow")
    
from unittest.mock import patch

@pytest.mark.skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available(), reason="Needs Accelerator")
@patch('src.data_generation.GenerationPipeline')
def test_proxy_model_loading(MockPipeline):
    """Test 1.3b: Proxy Model Loading MOCKED."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = MockPipeline.return_value
    pipeline.generate_traces.return_value = [{"trace": "mock trace 1", "answer": "1"}, {"trace": "mock trace 2", "answer": "2"}]
    
    questions = ["What is 2+2?"]
    ids = ["q1"]
    
    dataset = pipeline.generate_traces(questions, ids, k=2)
    assert len(dataset) == 2
    assert type(dataset[0]["trace"]) == str
