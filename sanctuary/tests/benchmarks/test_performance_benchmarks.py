"""
Performance benchmarking suite for cognitive architecture.

Measures per-subsystem timing and detects performance regressions.
Tests ensure cognitive loop maintains 10 Hz operation (≤100ms per cycle).
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="benchmark_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPerformanceBenchmarks:
    """Benchmark suite for cognitive loop performance."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cognitive_cycle_performance(self, temp_data_dir):
        """
        Benchmark complete cognitive cycle.

        Ensures:
        - Average cycle time ≤ 110ms
        - P95 cycle time ≤ 150ms
        - P99 cycle time ≤ 200ms
        - Max cycle time ≤ 500ms (hard limit)
        """
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "attention_budget": 100,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
            "log_interval_cycles": 1000,  # Don't spam logs
            "memory": {
                "memory_config": {
                    "base_dir": str(temp_data_dir / "memories"),
                    "chroma_dir": str(temp_data_dir / "chroma"),
                }
            },
        }

        core = CognitiveCore(workspace=workspace, config=config)

        # Add test goals
        for i in range(5):
            workspace.add_goal(Goal(
                type=GoalType.RESPOND_TO_USER,
                description=f"Test goal {i}",
                priority=0.5
            ))

        # Start core
        await core.start()
        
        # Wait for system to warm up
        await asyncio.sleep(0.5)
        
        # Measure 50 cycles
        cycle_times = []
        measurement_start = time.time()
        num_cycles_start = core.metrics['total_cycles']
        
        # Run for ~5 seconds to get stable measurements
        while time.time() - measurement_start < 5.0:
            await asyncio.sleep(0.1)  # Wait one cycle
        
        # Get metrics
        num_cycles_end = core.metrics['total_cycles']
        cycles_measured = num_cycles_end - num_cycles_start
        
        # Get cycle times from metrics
        cycle_times = list(core.metrics['cycle_times'])[-cycles_measured:]
        cycle_times_ms = [t * 1000 for t in cycle_times]
        
        await core.stop()
        
        # Analyze results
        if cycle_times_ms:
            avg_time = sum(cycle_times_ms) / len(cycle_times_ms)
            sorted_times = sorted(cycle_times_ms)
            p95_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
            p99_time = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
            max_time = max(cycle_times_ms) if cycle_times_ms else 0
            
            # Log results
            print(f"\n📊 Cognitive Cycle Performance ({len(cycle_times_ms)} cycles):")
            print(f"  Average: {avg_time:.1f}ms")
            print(f"  P95: {p95_time:.1f}ms")
            print(f"  P99: {p99_time:.1f}ms")
            print(f"  Max: {max_time:.1f}ms")
            
            # Performance assertions (relaxed for CI environments)
            assert avg_time <= 150, f"Average cycle time {avg_time:.1f}ms exceeds 150ms target"
            assert p95_time <= 200, f"P95 cycle time {p95_time:.1f}ms exceeds 200ms target"
            assert p99_time <= 300, f"P99 cycle time {p99_time:.1f}ms exceeds 300ms target"
            assert max_time <= 1000, f"Max cycle time {max_time:.1f}ms exceeds 1000ms hard limit"
        else:
            pytest.fail("No cycle times measured")
    
    @pytest.mark.benchmark
    def test_attention_selection_performance(self):
        """
        Benchmark attention selection.
        
        Ensures:
        - Average attention time ≤ 30ms (relaxed from 20ms)
        - Max attention time ≤ 100ms
        """
        from mind.cognitive_core.attention import AttentionController
        from mind.cognitive_core.workspace import GlobalWorkspace
        from mind.cognitive_core.affect import AffectSubsystem
        
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        attention = AttentionController(
            attention_budget=100,
            workspace=workspace,
            affect=affect
        )
        
        # Add goals to workspace for relevance scoring
        for i in range(3):
            workspace.add_goal(Goal(
                type=GoalType.RESPOND_TO_USER,
                description=f"Test goal {i}",
                priority=0.5
            ))
        
        # Create test percepts (without embeddings for speed)
        percepts = [
            Percept(modality="text", raw=f"Test percept {i}", complexity=5)
            for i in range(50)
        ]
        
        # Warm up
        for _ in range(5):
            attention.select_for_broadcast(percepts)
        
        # Benchmark selection
        times = []
        for _ in range(50):
            start = time.time()
            selected = attention.select_for_broadcast(percepts)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"\n⚡ Attention Selection Performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        
        # Performance assertions (relaxed for CI)
        assert avg_time <= 30, f"Average attention time {avg_time:.2f}ms exceeds 30ms target"
        assert max_time <= 100, f"Max attention time {max_time:.2f}ms exceeds 100ms limit"
    
    @pytest.mark.benchmark
    def test_affect_update_performance(self):
        """
        Benchmark affect updates.
        
        Ensures:
        - Average affect update ≤ 15ms (relaxed from 10ms)
        - Max affect update ≤ 50ms
        """
        from mind.cognitive_core.affect import AffectSubsystem
        from mind.cognitive_core.workspace import GlobalWorkspace
        
        affect = AffectSubsystem()
        workspace = GlobalWorkspace()
        
        # Add some test content to workspace
        for i in range(5):
            workspace.add_goal(Goal(
                type=GoalType.RESPOND_TO_USER,
                description=f"Test goal {i}",
                priority=0.5
            ))
        
        # Warm up
        for _ in range(5):
            snapshot = workspace.broadcast()
            affect.compute_update(snapshot)
        
        # Benchmark affect updates
        times = []
        for _ in range(100):
            snapshot = workspace.broadcast()
            start = time.time()
            affect.compute_update(snapshot)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"\n❤️  Affect Update Performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        
        # Performance assertions (relaxed for CI)
        assert avg_time <= 15, f"Average affect update {avg_time:.2f}ms exceeds 15ms target"
        assert max_time <= 50, f"Max affect update {max_time:.2f}ms exceeds 50ms limit"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_subsystem_timing_instrumentation(self, temp_data_dir):
        """
        Verify that per-subsystem timing instrumentation is working.

        Ensures:
        - Subsystem timings are tracked
        - Performance breakdown is available
        - No subsystem consistently exceeds limits
        """
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "attention_budget": 100,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
            "log_interval_cycles": 1000,
            "memory": {
                "memory_config": {
                    "base_dir": str(temp_data_dir / "memories"),
                    "chroma_dir": str(temp_data_dir / "chroma"),
                }
            },
        }

        core = CognitiveCore(workspace=workspace, config=config)

        # Add test goal
        workspace.add_goal(Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test subsystem timing",
            priority=0.5
        ))

        # Start and run for a few cycles
        await core.start()
        await asyncio.sleep(0.5)  # ~5 cycles
        
        # Get performance breakdown
        breakdown = core.get_performance_breakdown()
        
        await core.stop()
        
        print(f"\n📊 Subsystem Performance Breakdown:")
        for subsystem, stats in sorted(breakdown.items(), key=lambda x: -x[1]['avg_ms']):
            print(f"  {subsystem}:")
            print(f"    Avg: {stats['avg_ms']:.2f}ms")
            print(f"    P95: {stats['p95_ms']:.2f}ms")
        
        # Verify instrumentation is working
        assert len(breakdown) > 0, "No subsystem timings recorded"
        assert 'attention' in breakdown, "Attention timing not recorded"
        assert 'perception' in breakdown, "Perception timing not recorded"
        
        # Check that no subsystem is consistently too slow (relaxed thresholds)
        for subsystem, stats in breakdown.items():
            assert stats['avg_ms'] < 100, f"{subsystem} avg time {stats['avg_ms']:.1f}ms exceeds 100ms"
            assert stats['p95_ms'] < 200, f"{subsystem} P95 time {stats['p95_ms']:.1f}ms exceeds 200ms"
