import json
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Protocol
from src.tracker import TokenCostTracker
from src.config import FreightConfig, DEFAULT_CONFIG


@dataclass
class CompletionResult:
    """Encapsulates the generated text completion and token metadata."""
    text: str
    input_tokens: int
    output_tokens: int


class InferenceEngine(Protocol):
    """
    Execution seam for language model completions.
    Callers supply structured chat messages and optional context metadata.
    Implementations absorb model weight management, tokenization, device allocation,
    and token-level telemetry bookkeeping.
    """

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        call_context: Optional[Dict[str, Any]] = None
    ) -> CompletionResult:
        ...


class LocalQwenEngine:
    """
    Production adapter executing Qwen2.5-Instruct locally via PyTorch / HuggingFace.
    Encapsulates model initialization, greedy decoding, and token cost tracking.
    Uses class-level thread-safe singleton caching so 1.5B weights are loaded only once per process.
    """
    _lock = threading.Lock()
    _shared_models: Dict[tuple, Any] = {}
    _shared_tokenizers: Dict[tuple, Any] = {}
    _shared_model = None
    _shared_tokenizer = None

    def __init__(
        self,
        config: FreightConfig = DEFAULT_CONFIG,
        tracker: Optional[TokenCostTracker] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.config = config
        self.model_name = model_name or config.model_name
        self.tracker = tracker
        self.device = device or config.device
        self.seed = seed if seed is not None else config.seed

    @property
    def model(self):
        return LocalQwenEngine._shared_models.get((self.model_name, self.device))

    @property
    def tokenizer(self):
        return LocalQwenEngine._shared_tokenizers.get((self.model_name, self.device))

    def _ensure_model_loaded(self):
        key = (self.model_name, self.device)
        if LocalQwenEngine._shared_model is None:
            LocalQwenEngine._shared_models.pop(key, None)
        if LocalQwenEngine._shared_tokenizer is None:
            LocalQwenEngine._shared_tokenizers.pop(key, None)

        if key not in LocalQwenEngine._shared_models or key not in LocalQwenEngine._shared_tokenizers:
            with LocalQwenEngine._lock:
                if LocalQwenEngine._shared_model is None:
                    LocalQwenEngine._shared_models.pop(key, None)
                if LocalQwenEngine._shared_tokenizer is None:
                    LocalQwenEngine._shared_tokenizers.pop(key, None)

                if key not in LocalQwenEngine._shared_models or key not in LocalQwenEngine._shared_tokenizers:
                    import torch
                    import transformers
                    from transformers import AutoTokenizer, AutoModelForCausalLM

                    transformers.set_seed(self.seed)
                    print(f"[InferenceEngine] Loading {self.model_name} offline on {self.device} (seed={self.seed})...")
                    tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    mod = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        dtype=torch.float32,
                        device_map=self.device,
                        low_cpu_mem_usage=True,
                        local_files_only=True
                    )
                    mod.eval()
                    LocalQwenEngine._shared_models[key] = mod
                    LocalQwenEngine._shared_tokenizers[key] = tok
                    LocalQwenEngine._shared_model = mod
                    LocalQwenEngine._shared_tokenizer = tok
                    print("[InferenceEngine] Model loaded successfully.")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        call_context: Optional[Dict[str, Any]] = None
    ) -> CompletionResult:
        self._ensure_model_loaded()
        import torch

        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.device)
        input_token_count = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][input_token_count:]
        output_token_count = len(new_tokens)
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if self.tracker and call_context:
            self.tracker.record_call(
                route=str(call_context.get("route", "INFERENCE")),
                week_of=str(call_context.get("week_of", "")),
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                note_candidates=str(call_context.get("note_candidates", ""))
            )

        return CompletionResult(
            text=response_text,
            input_tokens=input_token_count,
            output_tokens=output_token_count
        )


class FakeInferenceEngine:
    """
    In-memory test adapter satisfying the InferenceEngine seam.
    Enables instant unit tests without loading PyTorch or weights.
    Matches responses based on prompt substrings or an explicit handler function.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        handler: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        default_response: str = "{}",
        tracker: Optional[TokenCostTracker] = None
    ):
        self.responses = responses or {}
        self.handler = handler
        self.default_response = default_response
        self.tracker = tracker
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        call_context: Optional[Dict[str, Any]] = None
    ) -> CompletionResult:
        full_content = " ".join([m.get("content", "") for m in messages])
        
        # Find matching canned response by keyword
        matched_text = None
        for key, resp in self.responses.items():
            if key in full_content:
                matched_text = resp
                break

        if matched_text is None:
            if self.handler:
                matched_text = self.handler(messages)
            else:
                matched_text = self.default_response

        input_token_count = max(1, len(full_content.split()))
        output_token_count = max(1, len(matched_text.split()))

        self.calls.append({
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "call_context": call_context,
            "response": matched_text
        })

        if self.tracker and call_context:
            self.tracker.record_call(
                route=str(call_context.get("route", "FAKE")),
                week_of=str(call_context.get("week_of", "")),
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                note_candidates=str(call_context.get("note_candidates", ""))
            )

        return CompletionResult(
            text=matched_text,
            input_tokens=input_token_count,
            output_tokens=output_token_count
        )
