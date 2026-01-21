"""
Dataset management for evaluation.

Features:
- Synthetic dataset generation
- Dataset versioning
- Train/test splits
- Human annotation tracking
- Regression test suites

Production features:
- Git-like versioning
- Metadata tracking
- Quality validation
- Export/import formats
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import asyncio

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_error,
)
from legal_assistant.core.exceptions import ValidationError
from legal_assistant.generation import LLMRouter

logger = get_logger(__name__)
settings = get_settings()


class DatasetType(str, Enum):
    """Dataset types."""
    
    SYNTHETIC = "synthetic"      # LLM-generated
    HUMAN_LABELED = "human"      # Human annotated
    MIXED = "mixed"              # Combination


class SplitType(str, Enum):
    """Dataset splits."""
    
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class DatasetSample:
    """Single dataset sample."""
    
    id: str
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    split: Optional[SplitType] = None
    annotator: Optional[str] = None
    annotation_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        if self.annotation_date:
            data["annotation_date"] = self.annotation_date.isoformat()
        if self.split:
            data["split"] = self.split.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetSample":
        """Create from dictionary."""
        if "annotation_date" in data and data["annotation_date"]:
            data["annotation_date"] = datetime.fromisoformat(
                data["annotation_date"]
            )
        if "split" in data and data["split"]:
            data["split"] = SplitType(data["split"])
        return cls(**data)


@dataclass
class DatasetMetadata:
    """Dataset metadata."""
    
    name: str
    version: str
    dataset_type: DatasetType
    sample_count: int
    created_at: datetime
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    parent_version: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["dataset_type"] = self.dataset_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetMetadata":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["dataset_type"] = DatasetType(data["dataset_type"])
        return cls(**data)


class DatasetManager:
    """
    Dataset management for evaluation.
    
    Features:
    - Create/load/save datasets
    - Version control
    - Train/val/test splits
    - Quality validation
    - Export formats (JSON, JSONL, CSV)
    
    Example:
        >>> manager = DatasetManager(data_dir="./datasets")
        >>> 
        >>> # Create dataset
        >>> samples = [DatasetSample(...), ...]
        >>> manager.save_dataset("legal_qa_v1", samples)
        >>> 
        >>> # Load dataset
        >>> loaded = manager.load_dataset("legal_qa_v1")
        >>> 
        >>> # Create splits
        >>> train, val, test = manager.create_splits(
        ...     loaded, train_ratio=0.7, val_ratio=0.15
        ... )
    """
    
    def __init__(self, data_dir: Path = Path("./data/evaluation")):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory for dataset storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("dataset_manager_initialized", data_dir=str(self.data_dir))
    
    def save_dataset(
        self,
        name: str,
        samples: List[DatasetSample],
        dataset_type: DatasetType = DatasetType.SYNTHETIC,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> DatasetMetadata:
        """
        Save dataset to disk.
        
        Args:
            name: Dataset name
            samples: Dataset samples
            dataset_type: Type of dataset
            description: Optional description
            tags: Optional tags
            version: Version string (auto-generated if None)
        
        Returns:
            DatasetMetadata
        
        Example:
            >>> metadata = manager.save_dataset(
            ...     "legal_qa",
            ...     samples,
            ...     description="Legal Q&A dataset",
            ...     tags=["contract", "law"],
            ... )
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)
        
        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            version=version,
            dataset_type=dataset_type,
            sample_count=len(samples),
            created_at=datetime.utcnow(),
            description=description,
            tags=tags,
            checksum=self._calculate_checksum(samples),
        )
        
        # Save to disk
        dataset_path = self._get_dataset_path(name, version)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": metadata.to_dict(),
                    "samples": [s.to_dict() for s in samples],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        
        logger.info(
            "dataset_saved",
            name=name,
            version=version,
            sample_count=len(samples),
            path=str(dataset_path),
        )
        
        return metadata
    
    def load_dataset(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> tuple[DatasetMetadata, List[DatasetSample]]:
        """
        Load dataset from disk.
        
        Args:
            name: Dataset name
            version: Version (None = latest)
        
        Returns:
            Tuple of (metadata, samples)
        
        Raises:
            ValidationError: If dataset not found
        """
        # Get version
        if version is None:
            version = self._get_latest_version(name)
        
        dataset_path = self._get_dataset_path(name, version)
        
        if not dataset_path.exists():
            raise ValidationError(
                message=f"Dataset not found: {name} v{version}",
                details={"name": name, "version": version},
            )
        
        # Load from disk
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        metadata = DatasetMetadata.from_dict(data["metadata"])
        samples = [DatasetSample.from_dict(s) for s in data["samples"]]
        
        logger.info(
            "dataset_loaded",
            name=name,
            version=version,
            sample_count=len(samples),
        )
        
        return metadata, samples
    
    def create_splits(
        self,
        samples: List[DatasetSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: Optional[float] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dict[SplitType, List[DatasetSample]]:
        """
        Create train/val/test splits.
        
        Args:
            samples: All samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio (auto-calculated if None)
            shuffle: Shuffle before splitting
            seed: Random seed
        
        Returns:
            Dict mapping split type to samples
        
        Example:
            >>> splits = manager.create_splits(
            ...     samples,
            ...     train_ratio=0.7,
            ...     val_ratio=0.15,
            ... )
            >>> train = splits[SplitType.TRAIN]
            >>> val = splits[SplitType.VAL]
            >>> test = splits[SplitType.TEST]
        """
        import random
        
        # Validate ratios
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio - val_ratio
        
        if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
            raise ValidationError(
                message="Split ratios must sum to 1.0",
                details={
                    "train": train_ratio,
                    "val": val_ratio,
                    "test": test_ratio,
                },
            )
        
        # Shuffle
        samples_copy = samples.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(samples_copy)
        
        # Calculate split indices
        n = len(samples_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_samples = samples_copy[:train_end]
        val_samples = samples_copy[train_end:val_end]
        test_samples = samples_copy[val_end:]
        
        # Assign split tags
        for s in train_samples:
            s.split = SplitType.TRAIN
        for s in val_samples:
            s.split = SplitType.VAL
        for s in test_samples:
            s.split = SplitType.TEST
        
        logger.info(
            "splits_created",
            train=len(train_samples),
            val=len(val_samples),
            test=len(test_samples),
        )
        
        return {
            SplitType.TRAIN: train_samples,
            SplitType.VAL: val_samples,
            SplitType.TEST: test_samples,
        }
    
    def export_dataset(
        self,
        name: str,
        version: Optional[str] = None,
        format: Literal["json", "jsonl", "csv"] = "json",
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export dataset to different format.
        
        Args:
            name: Dataset name
            version: Version (None = latest)
            format: Export format
            output_path: Output path (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        # Load dataset
        metadata, samples = self.load_dataset(name, version)
        
        # Generate output path
        if output_path is None:
            version_str = version or self._get_latest_version(name)
            output_path = self.data_dir / f"{name}_v{version_str}.{format}"
        
        # Export
        if format == "json":
            self._export_json(samples, output_path)
        elif format == "jsonl":
            self._export_jsonl(samples, output_path)
        elif format == "csv":
            self._export_csv(samples, output_path)
        
        logger.info(
            "dataset_exported",
            name=name,
            format=format,
            path=str(output_path),
        )
        
        return output_path
    
    def _export_json(self, samples: List[DatasetSample], path: Path):
        """Export to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in samples],
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    def _export_jsonl(self, samples: List[DatasetSample], path: Path):
        """Export to JSONL (one sample per line)."""
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    
    def _export_csv(self, samples: List[DatasetSample], path: Path):
        """Export to CSV."""
        import csv
        
        with open(path, "w", encoding="utf-8", newline="") as f:
            if not samples:
                return
            
            # Get field names
            fieldnames = list(samples[0].to_dict().keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample in samples:
                row = sample.to_dict()
                # Convert lists to JSON strings for CSV
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value, ensure_ascii=False)
                writer.writerow(row)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.
        
        Returns:
            List of dataset info dicts
        """
        datasets = []
        
        for dataset_dir in self.data_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            name = dataset_dir.name
            versions = []
            
            for version_file in dataset_dir.glob("v*.json"):
                version = version_file.stem
                
                try:
                    with open(version_file, "r") as f:
                        data = json.load(f)
                        metadata = DatasetMetadata.from_dict(data["metadata"])
                        versions.append({
                            "version": version,
                            "sample_count": metadata.sample_count,
                            "created_at": metadata.created_at.isoformat(),
                        })
                except Exception as e:
                    logger.warning(
                        "failed_to_read_dataset",
                        path=str(version_file),
                        error=str(e),
                    )
            
            if versions:
                datasets.append({
                    "name": name,
                    "versions": sorted(
                        versions,
                        key=lambda v: v["created_at"],
                        reverse=True,
                    ),
                })
        
        return datasets
    
    def validate_dataset(
        self,
        samples: List[DatasetSample],
    ) -> Dict[str, Any]:
        """
        Validate dataset quality.
        
        Checks:
        - No empty fields
        - Unique IDs
        - Context lengths
        - Answer lengths
        
        Returns:
            Validation report dict
        """
        issues = []
        warnings = []
        
        # Check unique IDs
        ids = [s.id for s in samples]
        if len(ids) != len(set(ids)):
            issues.append("Duplicate sample IDs found")
        
        # Check each sample
        for i, sample in enumerate(samples):
            # Empty question
            if not sample.question or not sample.question.strip():
                issues.append(f"Sample {i}: Empty question")
            
            # Empty answer
            if not sample.answer or not sample.answer.strip():
                issues.append(f"Sample {i}: Empty answer")
            
            # No contexts
            if not sample.contexts:
                warnings.append(f"Sample {i}: No contexts provided")
            
            # Very short answer
            if len(sample.answer) < 20:
                warnings.append(f"Sample {i}: Very short answer ({len(sample.answer)} chars)")
            
            # Very long answer
            if len(sample.answer) > 5000:
                warnings.append(f"Sample {i}: Very long answer ({len(sample.answer)} chars)")
        
        is_valid = len(issues) == 0
        
        report = {
            "valid": is_valid,
            "sample_count": len(samples),
            "issues": issues,
            "warnings": warnings,
        }
        
        logger.info(
            "dataset_validated",
            valid=is_valid,
            issue_count=len(issues),
            warning_count=len(warnings),
        )
        
        return report
    
    def _get_dataset_path(self, name: str, version: str) -> Path:
        """Get path to dataset file."""
        return self.data_dir / name / f"{version}.json"
    
    def _generate_version(self, name: str) -> str:
        """Generate next version number."""
        dataset_dir = self.data_dir / name
        
        if not dataset_dir.exists():
            return "v1"
        
        # Find max version
        versions = []
        for version_file in dataset_dir.glob("v*.json"):
            version_str = version_file.stem
            if version_str.startswith("v"):
                try:
                    num = int(version_str[1:])
                    versions.append(num)
                except ValueError:
                    pass
        
        next_version = max(versions) + 1 if versions else 1
        return f"v{next_version}"
    
    def _get_latest_version(self, name: str) -> str:
        """Get latest version of dataset."""
        dataset_dir = self.data_dir / name
        
        if not dataset_dir.exists():
            raise ValidationError(
                message=f"Dataset not found: {name}",
                details={"name": name},
            )
        
        versions = []
        for version_file in dataset_dir.glob("v*.json"):
            version_str = version_file.stem
            if version_str.startswith("v"):
                try:
                    num = int(version_str[1:])
                    versions.append((num, version_str))
                except ValueError:
                    pass
        
        if not versions:
            raise ValidationError(
                message=f"No versions found for dataset: {name}",
                details={"name": name},
            )
        
        return max(versions, key=lambda v: v[0])[1]
    
    def _calculate_checksum(self, samples: List[DatasetSample]) -> str:
        """Calculate dataset checksum."""
        data = json.dumps(
            [s.to_dict() for s in samples],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class SyntheticDataGenerator:
    """
    Generate synthetic evaluation datasets using LLM.
    
    Example:
        >>> generator = SyntheticDataGenerator()
        >>> samples = await generator.generate_qa_pairs(
        ...     domain="contract_law",
        ...     count=50,
        ...     difficulty="medium",
        ... )
    """
    
    GENERATION_PROMPT = """Ты — эксперт по созданию качественных датасетов для оценки юридических AI-систем.

Создай {count} пар вопрос-ответ на тему: {domain}

Требования:
- Вопросы должны быть разнообразными и реалистичными
- Ответы должны быть точными и развёрнутыми (3-5 предложений)
- Сложность: {difficulty}
- Включи юридические термины и ссылки на законы (если применимо)

Формат ответа (JSON):
```json
[
  {{
    "question": "Вопрос пользователя",
    "answer": "Развёрнутый ответ",
    "contexts": ["Релевантный контекст 1", "Релевантный контекст 2"],
    "metadata": {{"topic": "подтема", "difficulty": "easy|medium|hard"}}
  }},
  ...
]
```
"""
    
    def __init__(self, model: str = "gpt-4-turbo"):
        """Initialize synthetic data generator."""
        self.model = model
        self.llm_router = LLMRouter()
        
        logger.info("synthetic_generator_initialized", model=model)
    
    async def generate_qa_pairs(
        self,
        domain: str,
        count: int = 10,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
    ) -> List[DatasetSample]:
        """
        Generate synthetic Q&A pairs.
        
        Args:
            domain: Domain/topic (e.g., "contract_law", "labor_law")
            count: Number of pairs to generate
            difficulty: Difficulty level
        
        Returns:
            List of DatasetSamples
        """
        try:
            prompt = self.GENERATION_PROMPT.format(
                count=count,
                domain=domain,
                difficulty=difficulty,
            )
            
            logger.info(
                "generating_synthetic_data",
                domain=domain,
                count=count,
                difficulty=difficulty,
            )
            
            async with self.llm_router as router:
                response = await router.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.8,
                    max_tokens=4000,
                )
            
            # Parse JSON response
            import re
            json_match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content
            
            data = json.loads(json_str)
            
            # Convert to DatasetSamples
            samples = []
            for i, item in enumerate(data):
                sample = DatasetSample(
                    id=f"{domain}_{i}_{datetime.utcnow().timestamp()}",
                    question=item["question"],
                    answer=item["answer"],
                    contexts=item.get("contexts", []),
                    metadata=item.get("metadata", {}),
                )
                samples.append(sample)
            
            logger.info(
                "synthetic_data_generated",
                domain=domain,
                sample_count=len(samples),
            )
            
            return samples
        
        except Exception as e:
            track_error("evaluation", e)
            logger.exception("synthetic_generation_failed", domain=domain)
            return []


# Convenience functions

def save_dataset(
    name: str,
    samples: List[DatasetSample],
    **kwargs,
) -> DatasetMetadata:
    """Convenience function for saving dataset."""
    manager = DatasetManager()
    return manager.save_dataset(name, samples, **kwargs)


def load_dataset(
    name: str,
    version: Optional[str] = None,
) -> tuple[DatasetMetadata, List[DatasetSample]]:
    """Convenience function for loading dataset."""
    manager = DatasetManager()
    return manager.load_dataset(name, version)


async def generate_synthetic_data(
    domain: str,
    count: int = 10,
    difficulty: str = "medium",
) -> List[DatasetSample]:
    """Convenience function for synthetic generation."""
    generator = SyntheticDataGenerator()
    return await generator.generate_qa_pairs(domain, count, difficulty)
