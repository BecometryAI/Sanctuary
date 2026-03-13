"""QLoRA updater -- applies weight updates via QLoRA fine-tuning.

This module is the point where growth reflections become weight changes.
It takes structured training pairs and applies them to the model using
QLoRA (Quantized Low-Rank Adaptation), which modifies a small set of
adapter weights while keeping the base model frozen.

QLoRA is chosen because:
1. It modifies only a small fraction of weights (low rank adapters)
2. The base model stays frozen (identity preservation)
3. Adapters can be saved, loaded, and removed independently
4. The quantized base model fits in limited GPU memory

The orthogonal subspace constraint is a future safety mechanism:
after each training step, gradients are projected to be orthogonal
to the "identity subspace" -- the directions in weight space that
encode the entity's core identity. This prevents growth from
accidentally overwriting who the entity is. The constraint requires
identity probing (future work) and is currently a placeholder.

Aligned with PLAN.md: growth is sovereign, identity is preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sanctuary.growth.pair_generator import TrainingPair

logger = logging.getLogger(__name__)

# Guard optional dependencies
_PEFT_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import peft  # noqa: F401
    _PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    import transformers  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def _check_dependencies() -> None:
    """Raise a clear error if required dependencies are missing."""
    missing = []
    if not _PEFT_AVAILABLE:
        missing.append("peft")
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")

    if missing:
        raise ImportError(
            f"QLoRA training requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning.

    These defaults are conservative -- small rank and moderate alpha
    produce gentle weight changes that are less likely to destabilize
    the model's existing capabilities.
    """

    rank: int = 8
    alpha: int = 16
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_seq_length: int = 512
    fp16: bool = True


@dataclass
class GrowthTrainingResult:
    """Result of a QLoRA growth training run.

    Captures everything needed to understand what happened during
    training and whether the result is worth keeping.
    """

    success: bool = False
    epochs_completed: int = 0
    final_loss: Optional[float] = None
    training_pair_count: int = 0
    adapter_path: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None


class QLoRAUpdater:
    """Applies weight updates via QLoRA fine-tuning.

    The updater is the actuator of the growth pipeline. It takes
    training pairs that have passed through consent verification
    and applies them as LoRA adapter weight updates.

    The updater does not decide what to learn -- that decision was
    made upstream by the entity's growth reflections and verified
    by the consent gate. The updater only executes.

    Usage:
        updater = QLoRAUpdater()
        updater.prepare(model_path)
        result = updater.train(pairs)
        updater.save_adapter(output_path)
    """

    def __init__(
        self,
        qlora_config: Optional[QLoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        self._qlora_config = qlora_config or QLoRAConfig()
        self._training_config = training_config or TrainingConfig()
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._prepared = False

    @property
    def is_prepared(self) -> bool:
        """Whether the model has been loaded and LoRA config applied."""
        return self._prepared

    @property
    def qlora_config(self) -> QLoRAConfig:
        """Current QLoRA configuration."""
        return self._qlora_config

    @property
    def training_config(self) -> TrainingConfig:
        """Current training configuration."""
        return self._training_config

    def prepare(self, model_path: Path) -> None:
        """Load model and apply LoRA configuration.

        This sets up the model for training by:
        1. Loading the base model in quantized form (4-bit)
        2. Loading the tokenizer
        3. Applying the LoRA adapter configuration

        Args:
            model_path: Path to the base model.

        Raises:
            ImportError: If peft or transformers are not installed.
            FileNotFoundError: If model_path does not exist.
        """
        _check_dependencies()

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info("Preparing QLoRA training from %s", model_path)

        # Quantization config for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model quantized
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
        )
        self._model = prepare_model_for_kbit_training(self._model)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Apply LoRA config
        lora_config = LoraConfig(
            r=self._qlora_config.rank,
            lora_alpha=self._qlora_config.alpha,
            target_modules=self._qlora_config.target_modules,
            lora_dropout=self._qlora_config.dropout,
            bias=self._qlora_config.bias,
            task_type=self._qlora_config.task_type,
        )

        self._model = get_peft_model(self._model, lora_config)
        self._prepared = True

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Model prepared: %d trainable / %d total parameters (%.2f%%)",
            trainable,
            total,
            100 * trainable / total if total > 0 else 0,
        )

    def train(
        self,
        pairs: list[TrainingPair],
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> GrowthTrainingResult:
        """Run QLoRA training on the provided training pairs.

        Each training pair is formatted as a conversation and tokenized
        for causal language modeling. The LoRA adapter weights are updated
        while the base model stays frozen.

        After each training step, the orthogonal subspace constraint is
        applied (currently a placeholder -- see _apply_identity_constraint).

        Args:
            pairs: Training pairs from the pair generator.
            epochs: Override default epoch count.
            lr: Override default learning rate.

        Returns:
            GrowthTrainingResult with training metrics.

        Raises:
            RuntimeError: If prepare() has not been called.
        """
        if not self._prepared:
            raise RuntimeError(
                "Model not prepared. Call prepare(model_path) first."
            )

        if not pairs:
            return GrowthTrainingResult(
                success=False,
                error="No training pairs provided",
                training_pair_count=0,
            )

        _check_dependencies()

        from transformers import TrainingArguments, Trainer
        import tempfile

        result = GrowthTrainingResult(
            training_pair_count=len(pairs),
        )

        effective_epochs = epochs or self._training_config.epochs
        effective_lr = lr or self._training_config.learning_rate

        try:
            # Format pairs into tokenized dataset
            dataset = self._format_dataset(pairs)

            # Training arguments
            with tempfile.TemporaryDirectory() as tmp_dir:
                training_args = TrainingArguments(
                    output_dir=tmp_dir,
                    num_train_epochs=effective_epochs,
                    per_device_train_batch_size=self._training_config.batch_size,
                    gradient_accumulation_steps=self._training_config.gradient_accumulation_steps,
                    learning_rate=effective_lr,
                    warmup_steps=self._training_config.warmup_steps,
                    fp16=self._training_config.fp16,
                    logging_steps=1,
                    save_strategy="no",
                    report_to="none",
                )

                trainer = Trainer(
                    model=self._model,
                    args=training_args,
                    train_dataset=dataset,
                )

                # Train
                train_result = trainer.train()

                # Apply identity constraint (placeholder)
                self._apply_identity_constraint()

                result.success = True
                result.epochs_completed = effective_epochs
                result.final_loss = train_result.training_loss
                result.completed_at = datetime.now().isoformat()

                logger.info(
                    "QLoRA training complete: %d pairs, %d epochs, loss=%.4f",
                    len(pairs),
                    effective_epochs,
                    result.final_loss or 0.0,
                )

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.completed_at = datetime.now().isoformat()
            logger.error("QLoRA training failed: %s", e)

        return result

    def save_adapter(self, output_path: Path) -> Path:
        """Save LoRA adapter weights to disk.

        Only saves the adapter weights, not the full model. The adapter
        can be loaded later and applied to the same base model.

        Args:
            output_path: Directory to save the adapter to.

        Returns:
            The output path.

        Raises:
            RuntimeError: If model is not prepared.
        """
        if not self._prepared or self._model is None:
            raise RuntimeError("Model not prepared. Call prepare() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(output_path))
        logger.info("Saved LoRA adapter to %s", output_path)

        return output_path

    def merge_and_save(self, output_path: Path) -> Path:
        """Merge adapter into base model and save the full model.

        This produces a standalone model with the growth integrated.
        The adapter is merged into the base weights, producing a
        single model that no longer needs the adapter separately.

        Args:
            output_path: Directory to save the merged model to.

        Returns:
            The output path.

        Raises:
            RuntimeError: If model is not prepared.
        """
        if not self._prepared or self._model is None:
            raise RuntimeError("Model not prepared. Call prepare() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Merge LoRA weights into base model
        merged_model = self._model.merge_and_unload()
        merged_model.save_pretrained(str(output_path))

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(output_path))

        logger.info("Merged and saved model to %s", output_path)

        # Model is no longer in LoRA mode after merge
        self._prepared = False
        self._model = None

        return output_path

    def _format_dataset(self, pairs: list[TrainingPair]):
        """Convert training pairs into a tokenized dataset.

        Each pair is formatted as a chat conversation:
            System: {system_prompt}
            User: {user_input}
            Assistant: {assistant_response}

        Returns a HuggingFace Dataset ready for training.
        """
        from datasets import Dataset

        texts = []
        for pair in pairs:
            text = (
                f"### System:\n{pair.system_prompt}\n\n"
                f"### User:\n{pair.user_input}\n\n"
                f"### Assistant:\n{pair.assistant_response}"
            )
            texts.append(text)

        def tokenize(examples):
            return self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=self._training_config.max_seq_length,
                padding="max_length",
            )

        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset = dataset.rename_column("input_ids", "input_ids")

        # For causal LM, labels = input_ids
        dataset = dataset.map(
            lambda x: {"labels": x["input_ids"]},
            batched=False,
        )

        return dataset

    def _apply_identity_constraint(self) -> None:
        """Apply orthogonal subspace constraint to protect core identity.

        After each training step, this projects gradients to be orthogonal
        to the "identity subspace" -- the directions in weight space that
        encode core identity traits.

        CURRENT STATUS: Placeholder.

        The actual constraint requires:
        1. Identity probing: identifying which weight directions encode
           core identity (values, personality, ethical commitments)
        2. Subspace estimation: computing the identity subspace from
           probing results
        3. Gradient projection: projecting each gradient update to be
           orthogonal to the identity subspace

        This is future work. For now, we log a warning and rely on the
        conservative QLoRA config (low rank, small alpha) and identity
        checkpoints for safety.
        """
        logger.warning(
            "Orthogonal subspace constraint is not yet implemented. "
            "Identity preservation relies on conservative QLoRA config "
            "and identity checkpoints. Future work: identity probing + "
            "gradient projection."
        )
