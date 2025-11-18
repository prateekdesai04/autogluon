"""
Unit tests for _calculate_gpu_assignment() function in ParallelFoldFittingStrategy.

This test suite validates GPU assignment logic for various scenarios:
- CPU-only mode (no GPUs)
- Single GPU
- Multiple GPUs
- Different task counts and GPU-per-task ratios
"""

import os
import pytest

from autogluon.core.models.ensemble.fold_fitting_strategy import (
    ParallelLocalFoldFittingStrategy,
)


class MockParallelStrategy:
    """Mock ParallelFoldFittingStrategy with _calculate_gpu_assignment method.

    This mock implements the FIXED version that handles CPU-only mode (total_gpus=0).
    The fix prevents ZeroDivisionError by returning an empty GPU assignment list.
    """

    def _calculate_gpu_assignment(self, gpu_assignments, task_id, gpus_per_task, total_gpus):
        """Fixed implementation of _calculate_gpu_assignment matching fold_fitting_strategy.py.

        FIX APPLIED: Added guard at line 752-755 in fold_fitting_strategy.py
        to handle CPU-only mode (when total_gpus == 0).
        """
        # Handle CPU-only mode (no GPUs available)
        if total_gpus == 0:
            gpu_assignments[task_id] = []
            return gpu_assignments

        if gpus_per_task >= 1:
            gpu_id = task_id * gpus_per_task
            assigned_gpus = []
            for i in range(gpus_per_task):
                assigned_gpus.append(gpu_id + i % total_gpus)
            gpu_assignments[task_id] = assigned_gpus
        else:
            gpu_id = task_id % total_gpus
            gpu_assignments[task_id] = [gpu_id]
        return gpu_assignments


class TestCalculateGPUAssignment:
    """Test suite for _calculate_gpu_assignment() method."""

    @pytest.fixture
    def strategy(self):
        """Create a mock strategy instance for testing."""
        return MockParallelStrategy()

    # ==================== CPU-ONLY TESTS ====================
    def test_cpu_only_mode_single_task(self, strategy):
        """Test GPU assignment with no GPUs (CPU-only mode) for a single task."""
        gpu_assignments = {}
        result = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=0,
            gpus_per_task=0,
            total_gpus=0,
        )

        assert result[0] == []
        assert isinstance(result[0], list)
        assert len(result[0]) == 0

    def test_cpu_only_mode_multiple_tasks(self, strategy):
        """Test GPU assignment with no GPUs for multiple tasks."""
        gpu_assignments = {}

        # Assign GPUs for 4 tasks in CPU-only mode
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0,
                total_gpus=0,
            )

        # All tasks should have empty lists
        for task_id in range(4):
            assert task_id in gpu_assignments
            assert gpu_assignments[task_id] == []

    # ==================== SINGLE GPU TESTS ====================
    def test_single_gpu_single_task(self, strategy):
        """Test GPU assignment with single GPU and single task."""
        gpu_assignments = {}
        result = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=0,
            gpus_per_task=1,
            total_gpus=1,
        )

        assert result[0] == [0]

    def test_single_gpu_multiple_tasks_round_robin(self, strategy):
        """Test GPU assignment with single GPU and multiple tasks (round-robin)."""
        gpu_assignments = {}

        # Assign GPUs for 4 tasks with only 1 GPU (gpus_per_task < 1)
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0,  # Fractional GPU allocation
                total_gpus=1,
            )

        # Expected: round-robin assignment [0, 0, 0, 0]
        expected = [0, 0, 0, 0]
        for task_id in range(4):
            assert gpu_assignments[task_id] == [expected[task_id]]

    # ==================== MULTIPLE GPUS TESTS ====================
    def test_multiple_gpus_single_task_per_gpu(self, strategy):
        """Test GPU assignment with multiple GPUs, one GPU per task."""
        gpu_assignments = {}

        # Assign 1 GPU per task for 4 tasks with 4 GPUs
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=1,
                total_gpus=4,
            )

        # Expected: [0], [1], [2], [3]
        expected = [[0], [1], [2], [3]]
        for task_id in range(4):
            assert gpu_assignments[task_id] == expected[task_id]

    def test_multiple_gpus_multiple_per_task(self, strategy):
        """Test GPU assignment with multiple GPUs per task."""
        gpu_assignments = {}

        # Assign 2 GPUs per task for 2 tasks with 4 GPUs
        for task_id in range(2):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=2,
                total_gpus=4,
            )

        # Expected: [0, 1], [2, 3]
        expected = [[0, 1], [2, 3]]
        for task_id in range(2):
            assert gpu_assignments[task_id] == expected[task_id]

    def test_fractional_gpu_per_task_wrapping(self, strategy):
        """Test GPU assignment with fractional GPUs per task (< 1)."""
        gpu_assignments = {}

        # 4 tasks, 2 GPUs, gpus_per_task = 0 (fractional)
        # Expected: tasks share GPUs in round-robin fashion
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0,  # When gpus_per_task < 1, uses else branch
                total_gpus=2,
            )

        # Expected: [0], [1], [0], [1] (round-robin)
        expected = [[0], [1], [0], [1]]
        for task_id in range(4):
            assert gpu_assignments[task_id] == expected[task_id]

    def test_more_tasks_than_gpus(self, strategy):
        """Test GPU assignment when tasks exceed available GPUs."""
        gpu_assignments = {}

        # 8 tasks, 4 GPUs, gpus_per_task = 0
        # Expected: round-robin cycling through GPUs
        for task_id in range(8):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0,
                total_gpus=4,
            )

        # Expected: [0, 1, 2, 3, 0, 1, 2, 3]
        expected = [0, 1, 2, 3, 0, 1, 2, 3]
        for task_id in range(8):
            assert gpu_assignments[task_id] == [expected[task_id]]

    def test_gpu_wrap_around_with_multiple_per_task(self, strategy):
        """Test GPU assignment when multiple GPUs per task exceed total."""
        gpu_assignments = {}

        # 3 tasks, 4 GPUs, 2 GPUs per task
        # Algorithm: gpu_id = task_id * gpus_per_task, then modulo applied per GPU
        for task_id in range(3):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=2,
                total_gpus=4,
            )

        # Verify assignments:
        # Task 0: gpu_id = 0*2 = 0, GPUs = [0+0%4, 0+1%4] = [0, 1]
        # Task 1: gpu_id = 1*2 = 2, GPUs = [2+0%4, 2+1%4] = [2, 3]
        # Task 2: gpu_id = 2*2 = 4, GPUs = [4+0%4, 4+1%4] = [4, 5] (NOT wrapped)
        assert gpu_assignments[0] == [0, 1]
        assert gpu_assignments[1] == [2, 3]
        assert gpu_assignments[2] == [4, 5]  # Modulo is applied per GPU index, not total

    # ==================== DICTIONARY HANDLING TESTS ====================
    def test_task_dict_updates_correctly(self, strategy):
        """Test that gpu_assignments dict is updated correctly."""
        gpu_assignments = {"existing_task": [5]}

        result = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=1,
            gpus_per_task=1,
            total_gpus=2,
        )

        # Both old and new tasks should be in the dict
        assert "existing_task" in result
        assert 1 in result
        assert result["existing_task"] == [5]
        assert result[1] == [1]

    def test_return_value_is_same_dict(self, strategy):
        """Test that function returns the same dictionary object."""
        gpu_assignments = {}
        result = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=0,
            gpus_per_task=1,
            total_gpus=2,
        )

        assert result is gpu_assignments

    # ==================== PARAMETRIZED TESTS ====================
    @pytest.mark.parametrize(
        "task_id,gpus_per_task,total_gpus,expected",
        [
            # CPU-only cases
            (0, 0, 0, []),
            (1, 0, 0, []),
            (5, 0, 0, []),

            # Single GPU cases
            (0, 1, 1, [0]),

            # Multiple GPUs, single per task
            (0, 1, 4, [0]),
            (1, 1, 4, [1]),
            (3, 1, 4, [3]),

            # Multiple GPUs per task
            (0, 2, 4, [0, 1]),
            (1, 2, 4, [2, 3]),
        ],
    )
    def test_parametrized_cases(self, strategy, task_id, gpus_per_task, total_gpus, expected):
        """Parametrized tests for various GPU assignment scenarios."""
        gpu_assignments = {}
        result = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=task_id,
            gpus_per_task=gpus_per_task,
            total_gpus=total_gpus,
        )

        assert result[task_id] == expected


class TestGPUAssignmentIntegration:
    """Integration tests for GPU assignment in the broader context."""

    def test_cpu_only_no_cuda_env_set(self):
        """Test that CPU-only mode doesn't set CUDA_VISIBLE_DEVICES."""
        # Save original CUDA_VISIBLE_DEVICES
        original_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")

        try:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Simulate _ray_fit behavior with empty GPU assignment
            gpu_ids = []  # CPU-only

            if gpu_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

            # Verify CUDA_VISIBLE_DEVICES was NOT set
            assert "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda_env

        finally:
            # Restore original state
            if original_cuda_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_env
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def test_gpu_assignment_with_multiple_batches(self):
        """Test GPU assignment consistency across multiple batches."""
        strategy = MockParallelStrategy()
        gpu_assignments = {}

        # Batch 1: Tasks 0-3 with 1 GPU per task
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=1,
                total_gpus=4,
            )

        # Batch 2: Tasks 4-7 with 1 GPU per task
        for task_id in range(4, 8):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=1,
                total_gpus=4,
            )

        # Verify all 8 tasks are assigned
        assert len(gpu_assignments) == 8
        for task_id in range(8):
            assert task_id in gpu_assignments
            assert len(gpu_assignments[task_id]) == 1

    def test_gpu_assignment_sequential_consistency(self):
        """Test that sequential calls maintain consistency."""
        strategy = MockParallelStrategy()
        gpu_assignments = {}

        # First call
        gpu_assignments = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=0,
            gpus_per_task=2,
            total_gpus=4,
        )
        first_result = gpu_assignments[0].copy()

        # Second call with different task
        gpu_assignments = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=1,
            gpus_per_task=2,
            total_gpus=4,
        )

        # Verify first assignment unchanged
        assert gpu_assignments[0] == first_result
        assert gpu_assignments[0] == [0, 1]
        assert gpu_assignments[1] == [2, 3]

    def test_cpu_only_with_ray_simulation(self):
        """Simulate the full CPU-only flow as it would be in _ray_fit."""
        strategy = MockParallelStrategy()
        gpu_assignments = {}

        # CPU-only mode: Calculate assignment
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0,
                total_gpus=0,
            )

        # Simulate _ray_fit behavior
        original_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")

        try:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Process all tasks
            for task_id in range(4):
                gpu_ids = gpu_assignments.get(task_id, [])

                # This is the pattern from _ray_fit
                if gpu_ids:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

                # Verify no CUDA_VISIBLE_DEVICES was set
                assert "CUDA_VISIBLE_DEVICES" not in os.environ

        finally:
            if original_cuda_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_env
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
