"""
Metrics tracking and aggregation for evaluation.

Features:
- Metric history tracking
- Performance trends
- Regression detection
- Dashboard data export
- Alerting on degradation

Production features:
- Time-series storage
- Aggregation by time period
- Comparison across versions
- Export to monitoring systems
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_error,
)
from legal_assistant.core.exceptions import ValidationError

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class MetricPoint:
    """Single metric measurement."""
    
    name: str
    value: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MetricPoint":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata"),
        )


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    
    name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    p95: float
    p99: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrendAnalysis:
    """Trend analysis result."""
    
    metric_name: str
    trend: Literal["improving", "stable", "degrading"]
    change_percentage: float
    recent_mean: float
    baseline_mean: float
    confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsTracker:
    """
    Track and analyze evaluation metrics over time.
    
    Features:
    - Store metric history
    - Calculate statistics
    - Detect trends
    - Export for dashboards
    - Alert on degradation
    
    Example:
        >>> tracker = MetricsTracker()
        >>> 
        >>> # Record metrics
        >>> tracker.record("faithfulness", 0.85, metadata={"model": "v1"})
        >>> tracker.record("answer_relevancy", 0.78)
        >>> 
        >>> # Get statistics
        >>> stats = tracker.get_statistics("faithfulness")
        >>> print(f"Mean: {stats.mean:.3f}")
        >>> 
        >>> # Detect trends
        >>> trend = tracker.analyze_trend("faithfulness", days=7)
        >>> if trend.trend == "degrading":
        ...     print("ALERT: Performance degrading!")
    """
    
    def __init__(
        self,
        storage_path: Path = Path("./data/evaluation/metrics"),
    ):
        """
        Initialize metrics tracker.
        
        Args:
            storage_path: Path to metrics storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._load_cache()
        
        logger.info(
            "metrics_tracker_initialized",
            storage_path=str(self.storage_path),
        )
    
    def record(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            timestamp: Timestamp (None = now)
            metadata: Additional metadata
        
        Example:
            >>> tracker.record(
            ...     "context_precision",
            ...     0.92,
            ...     metadata={"dataset": "legal_qa_v1", "model": "gpt-4"}
            ... )
        """
        timestamp = timestamp or datetime.utcnow()
        
        point = MetricPoint(
            name=metric_name,
            value=value,
            timestamp=timestamp,
            metadata=metadata,
        )
        
        # Add to cache
        self._cache[metric_name].append(point)
        
        # Persist to disk
        self._append_to_storage(point)
        
        logger.debug(
            "metric_recorded",
            name=metric_name,
            value=value,
        )
    
    def record_batch(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record multiple metrics at once.
        
        Args:
            metrics: Dict of metric_name -> value
            timestamp: Timestamp (None = now)
            metadata: Metadata for all metrics
        
        Example:
            >>> tracker.record_batch({
            ...     "context_precision": 0.85,
            ...     "context_recall": 0.78,
            ...     "faithfulness": 0.92,
            ...     "answer_relevancy": 0.81,
            ... })
        """
        for name, value in metrics.items():
            self.record(name, value, timestamp, metadata)
    
    def get_history(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[MetricPoint]:
        """
        Get metric history.
        
        Args:
            metric_name: Metric name
            start_time: Start time filter (None = beginning)
            end_time: End time filter (None = now)
            limit: Max points to return (None = all)
        
        Returns:
            List of MetricPoints
        """
        points = self._cache.get(metric_name, [])
        
        # Filter by time
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        # Sort by timestamp
        points = sorted(points, key=lambda p: p.timestamp)
        
        # Limit
        if limit:
            points = points[-limit:]
        
        return points
    
    def get_statistics(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[MetricSummary]:
        """
        Calculate statistics for a metric.
        
        Args:
            metric_name: Metric name
            start_time: Start time (None = all)
            end_time: End time (None = now)
        
        Returns:
            MetricSummary or None if no data
        
        Example:
            >>> stats = tracker.get_statistics("faithfulness", days=7)
            >>> print(f"Mean: {stats.mean:.3f} ± {stats.std_dev:.3f}")
            >>> print(f"P95: {stats.p95:.3f}")
        """
        points = self.get_history(metric_name, start_time, end_time)
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        # Calculate percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        p95_idx = int(0.95 * n)
        p99_idx = int(0.99 * n)
        
        return MetricSummary(
            name=metric_name,
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            p95=sorted_values[p95_idx] if p95_idx < n else sorted_values[-1],
            p99=sorted_values[p99_idx] if p99_idx < n else sorted_values[-1],
        )
    
    def analyze_trend(
        self,
        metric_name: str,
        recent_days: int = 7,
        baseline_days: int = 30,
        degradation_threshold: float = 0.05,
    ) -> Optional[TrendAnalysis]:
        """
        Analyze metric trend.
        
        Args:
            metric_name: Metric name
            recent_days: Days for recent period
            baseline_days: Days for baseline period
            degradation_threshold: % drop to consider degradation
        
        Returns:
            TrendAnalysis or None if insufficient data
        
        Example:
            >>> trend = tracker.analyze_trend("faithfulness", recent_days=7)
            >>> if trend.trend == "degrading":
            ...     print(f"Alert: {trend.metric_name} degraded by {trend.change_percentage:.1f}%")
        """
        now = datetime.utcnow()
        
        # Get recent and baseline periods
        recent_start = now - timedelta(days=recent_days)
        baseline_start = now - timedelta(days=baseline_days)
        baseline_end = now - timedelta(days=recent_days)
        
        recent_points = self.get_history(metric_name, recent_start, now)
        baseline_points = self.get_history(metric_name, baseline_start, baseline_end)
        
        if not recent_points or not baseline_points:
            return None
        
        # Calculate means
        recent_mean = statistics.mean([p.value for p in recent_points])
        baseline_mean = statistics.mean([p.value for p in baseline_points])
        
        # Calculate change
        change_percentage = ((recent_mean - baseline_mean) / baseline_mean) * 100
        
        # Determine trend
        if change_percentage > degradation_threshold * 100:
            trend = "improving"
        elif change_percentage < -degradation_threshold * 100:
            trend = "degrading"
        else:
            trend = "stable"
        
        # Calculate confidence (based on sample size and std dev)
        recent_std = statistics.stdev([p.value for p in recent_points]) if len(recent_points) > 1 else 0.0
        baseline_std = statistics.stdev([p.value for p in baseline_points]) if len(baseline_points) > 1 else 0.0
        
        # Simple confidence based on sample size and consistency
        confidence = min(
            len(recent_points) / 50,  # More samples = higher confidence
            1.0 - (recent_std + baseline_std) / 2,  # Lower variance = higher confidence
        )
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(
            "trend_analyzed",
            metric=metric_name,
            trend=trend,
            change_pct=round(change_percentage, 2),
        )
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend=trend,
            change_percentage=change_percentage,
            recent_mean=recent_mean,
            baseline_mean=baseline_mean,
            confidence=confidence,
        )
    
    def compare_versions(
        self,
        metric_name: str,
        version1_metadata: Dict[str, Any],
        version2_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare metric across two versions.
        
        Args:
            metric_name: Metric to compare
            version1_metadata: Metadata filter for version 1
            version2_metadata: Metadata filter for version 2
        
        Returns:
            Comparison dict
        
        Example:
            >>> comparison = tracker.compare_versions(
            ...     "faithfulness",
            ...     version1_metadata={"model": "v1"},
            ...     version2_metadata={"model": "v2"},
            ... )
            >>> print(f"Improvement: {comparison['improvement_pct']:.1f}%")
        """
        # Filter points by metadata
        all_points = self._cache.get(metric_name, [])
        
        v1_points = [
            p for p in all_points
            if self._matches_metadata(p.metadata, version1_metadata)
        ]
        
        v2_points = [
            p for p in all_points
            if self._matches_metadata(p.metadata, version2_metadata)
        ]
        
        if not v1_points or not v2_points:
            return {
                "error": "Insufficient data for comparison",
                "v1_count": len(v1_points),
                "v2_count": len(v2_points),
            }
        
        # Calculate statistics
        v1_mean = statistics.mean([p.value for p in v1_points])
        v2_mean = statistics.mean([p.value for p in v2_points])
        
        improvement = v2_mean - v1_mean
        improvement_pct = (improvement / v1_mean) * 100
        
        return {
            "metric_name": metric_name,
            "v1_mean": v1_mean,
            "v2_mean": v2_mean,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "v1_count": len(v1_points),
            "v2_count": len(v2_points),
        }
    
    def export_for_dashboard(
        self,
        metrics: Optional[List[str]] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Export metrics for dashboard/visualization.
        
        Args:
            metrics: Metrics to export (None = all)
            days: Days of history
        
        Returns:
            Dashboard-ready data dict
        
        Example:
            >>> data = tracker.export_for_dashboard(days=7)
            >>> # Send to Grafana/custom dashboard
        """
        start_time = datetime.utcnow() - timedelta(days=days)
        
        # Get all metrics if not specified
        if metrics is None:
            metrics = list(self._cache.keys())
        
        dashboard_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "period_days": days,
            "metrics": {},
        }
        
        for metric_name in metrics:
            history = self.get_history(metric_name, start_time)
            stats = self.get_statistics(metric_name, start_time)
            trend = self.analyze_trend(metric_name, days // 2, days)
            
            if stats:
                dashboard_data["metrics"][metric_name] = {
                    "current_value": history[-1].value if history else None,
                    "statistics": stats.to_dict(),
                    "trend": trend.to_dict() if trend else None,
                    "history": [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "value": p.value,
                        }
                        for p in history
                    ],
                }
        
        logger.info(
            "dashboard_data_exported",
            metric_count=len(dashboard_data["metrics"]),
            days=days,
        )
        
        return dashboard_data
    
    def check_alerts(
        self,
        thresholds: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """
        Check for metric alerts based on thresholds.
        
        Args:
            thresholds: Dict of metric_name -> {"min": x, "max": y}
        
        Returns:
            List of triggered alerts
        
        Example:
            >>> alerts = tracker.check_alerts({
            ...     "faithfulness": {"min": 0.8},
            ...     "answer_relevancy": {"min": 0.7},
            ... })
            >>> for alert in alerts:
            ...     print(f"ALERT: {alert['message']}")
        """
        alerts = []
        
        for metric_name, threshold in thresholds.items():
            # Get latest value
            history = self.get_history(metric_name, limit=1)
            
            if not history:
                continue
            
            latest = history[-1]
            
            # Check min threshold
            if "min" in threshold and latest.value < threshold["min"]:
                alerts.append({
                    "metric": metric_name,
                    "type": "below_minimum",
                    "value": latest.value,
                    "threshold": threshold["min"],
                    "message": f"{metric_name} ({latest.value:.3f}) below minimum ({threshold['min']:.3f})",
                    "timestamp": latest.timestamp.isoformat(),
                })
            
            # Check max threshold
            if "max" in threshold and latest.value > threshold["max"]:
                alerts.append({
                    "metric": metric_name,
                    "type": "above_maximum",
                    "value": latest.value,
                    "threshold": threshold["max"],
                    "message": f"{metric_name} ({latest.value:.3f}) above maximum ({threshold['max']:.3f})",
                    "timestamp": latest.timestamp.isoformat(),
                })
        
        if alerts:
            logger.warning(
                "metric_alerts_triggered",
                alert_count=len(alerts),
            )
        
        return alerts
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all tracked metric names."""
        return list(self._cache.keys())
    
    def clear_cache(self):
        """Clear in-memory cache (data persisted on disk)."""
        self._cache.clear()
        logger.info("metrics_cache_cleared")
    
    def _load_cache(self):
        """Load metrics from storage into cache."""
        try:
            for metric_file in self.storage_path.glob("*.jsonl"):
                metric_name = metric_file.stem
                
                with open(metric_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            point = MetricPoint.from_dict(json.loads(line))
                            self._cache[metric_name].append(point)
            
            logger.info(
                "metrics_cache_loaded",
                metric_count=len(self._cache),
                total_points=sum(len(points) for points in self._cache.values()),
            )
        
        except Exception as e:
            track_error("evaluation", e)
            logger.exception("failed_to_load_metrics_cache")
    
    def _append_to_storage(self, point: MetricPoint):
        """Append metric point to storage."""
        try:
            metric_file = self.storage_path / f"{point.name}.jsonl"
            
            with open(metric_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(point.to_dict(), ensure_ascii=False) + "\n")
        
        except Exception as e:
            track_error("evaluation", e)
            logger.exception("failed_to_persist_metric", metric=point.name)
    
    def _matches_metadata(
        self,
        point_metadata: Optional[Dict],
        filter_metadata: Dict,
    ) -> bool:
        """Check if point metadata matches filter."""
        if not point_metadata:
            return False
        
        for key, value in filter_metadata.items():
            if point_metadata.get(key) != value:
                return False
        
        return True


class RegressionTestSuite:
    """
    Regression test suite for model changes.
    
    Example:
        >>> suite = RegressionTestSuite()
        >>> suite.add_test("faithfulness", min_value=0.8)
        >>> suite.add_test("answer_relevancy", min_value=0.7)
        >>> 
        >>> results = suite.run(metrics_tracker)
        >>> if not results["all_passed"]:
        ...     print("Regression detected!")
    """
    
    def __init__(self):
        """Initialize regression test suite."""
        self.tests: List[Dict[str, Any]] = []
        
        logger.info("regression_suite_initialized")
    
    def add_test(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_degradation_pct: Optional[float] = None,
        baseline_days: int = 30,
    ):
        """
        Add a regression test.
        
        Args:
            metric_name: Metric to test
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            max_degradation_pct: Max % degradation from baseline
            baseline_days: Days for baseline calculation
        """
        self.tests.append({
            "metric_name": metric_name,
            "min_value": min_value,
            "max_value": max_value,
            "max_degradation_pct": max_degradation_pct,
            "baseline_days": baseline_days,
        })
        
        logger.debug("regression_test_added", metric=metric_name)
    
    def run(self, tracker: MetricsTracker) -> Dict[str, Any]:
        """
        Run all regression tests.
        
        Args:
            tracker: MetricsTracker instance
        
        Returns:
            Test results dict
        """
        results = {
            "total_tests": len(self.tests),
            "passed": 0,
            "failed": 0,
            "all_passed": True,
            "failures": [],
        }
        
        for test in self.tests:
            metric_name = test["metric_name"]
            
            # Get latest value
            history = tracker.get_history(metric_name, limit=1)
            
            if not history:
                results["failures"].append({
                    "metric": metric_name,
                    "reason": "No data available",
                })
                results["failed"] += 1
                results["all_passed"] = False
                continue
            
            current_value = history[-1].value
            
            # Check min threshold
            if test["min_value"] is not None:
                if current_value < test["min_value"]:
                    results["failures"].append({
                        "metric": metric_name,
                        "reason": f"Value {current_value:.3f} below minimum {test['min_value']:.3f}",
                        "current": current_value,
                        "threshold": test["min_value"],
                    })
                    results["failed"] += 1
                    results["all_passed"] = False
                    continue
            
            # Check max threshold
            if test["max_value"] is not None:
                if current_value > test["max_value"]:
                    results["failures"].append({
                        "metric": metric_name,
                        "reason": f"Value {current_value:.3f} above maximum {test['max_value']:.3f}",
                        "current": current_value,
                        "threshold": test["max_value"],
                    })
                    results["failed"] += 1
                    results["all_passed"] = False
                    continue
            
            # Check degradation
            if test["max_degradation_pct"] is not None:
                trend = tracker.analyze_trend(
                    metric_name,
                    recent_days=7,
                    baseline_days=test["baseline_days"],
                )
                
                if trend and trend.change_percentage < -test["max_degradation_pct"]:
                    results["failures"].append({
                        "metric": metric_name,
                        "reason": f"Degraded by {-trend.change_percentage:.1f}% (max: {test['max_degradation_pct']}%)",
                        "degradation": trend.change_percentage,
                        "threshold": test["max_degradation_pct"],
                    })
                    results["failed"] += 1
                    results["all_passed"] = False
                    continue
            
            # Test passed
            results["passed"] += 1
        
        logger.info(
            "regression_tests_completed",
            passed=results["passed"],
            failed=results["failed"],
        )
        
        return results


# Convenience functions

def record_metric(metric_name: str, value: float, **kwargs):
    """Convenience function for recording metric."""
    tracker = MetricsTracker()
    tracker.record(metric_name, value, **kwargs)


def get_metric_stats(metric_name: str, days: int = 30) -> Optional[MetricSummary]:
    """Convenience function for getting statistics."""
    tracker = MetricsTracker()
    start_time = datetime.utcnow() - timedelta(days=days)
    return tracker.get_statistics(metric_name, start_time)
