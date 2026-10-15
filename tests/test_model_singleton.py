import pytest
from unittest.mock import patch, MagicMock
from src.inference import LocalQwenEngine


def test_local_qwen_engine_does_not_load_model_multiple_times_across_instances():
    """
    Asserts that multiple instances of LocalQwenEngine share the loaded model weights
    and tokenizer rather than re-loading 1.5B parameters into CPU RAM on every instance.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_tok_loader, \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_model_loader:

        # Reset any shared state if it exists (for test isolation)
        if hasattr(LocalQwenEngine, "_shared_model"):
            LocalQwenEngine._shared_model = None
        if hasattr(LocalQwenEngine, "_shared_tokenizer"):
            LocalQwenEngine._shared_tokenizer = None

        engine1 = LocalQwenEngine(model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
        engine2 = LocalQwenEngine(model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cpu")

        # First instance loads the model
        engine1._ensure_model_loaded()
        assert mock_model_loader.call_count == 1, "First engine should load model weights"
        assert mock_tok_loader.call_count == 1, "First engine should load tokenizer"

        # Second instance MUST reuse the shared weights, NOT load them again
        engine2._ensure_model_loaded()
        assert mock_model_loader.call_count == 1, (
            f"Expected model to be loaded exactly once across instances, "
            f"but from_pretrained was called {mock_model_loader.call_count} times!"
        )
        assert mock_tok_loader.call_count == 1, (
            f"Expected tokenizer to be loaded exactly once across instances, "
            f"but tokenizer was loaded {mock_tok_loader.call_count} times!"
        )

        assert engine1.model is engine2.model
        assert engine1.tokenizer is engine2.tokenizer


def test_local_qwen_engine_thread_safe_concurrent_loading():
    """
    Asserts that multiple concurrent threads attempting to load the model
    simultaneously only trigger from_pretrained once due to proper lock synchronization.
    """
    import threading
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_tok_loader, \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_model_loader:

        if hasattr(LocalQwenEngine, "_shared_model"):
            LocalQwenEngine._shared_model = None
        if hasattr(LocalQwenEngine, "_shared_tokenizer"):
            LocalQwenEngine._shared_tokenizer = None

        engines = [LocalQwenEngine(model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cpu") for _ in range(5)]
        barrier = threading.Barrier(5)

        def worker(eng):
            barrier.wait()
            eng._ensure_model_loaded()

        threads = [threading.Thread(target=worker, args=(eng,)) for eng in engines]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mock_model_loader.call_count == 1, (
            f"Expected exactly 1 model load under concurrent access, got {mock_model_loader.call_count}"
        )

