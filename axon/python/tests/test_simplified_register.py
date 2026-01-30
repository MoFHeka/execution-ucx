"""Tests for simplified register_function_raw API with automatic type inference."""

import pytest
from typing import Tuple, List

# Import test_utils first to setup module path
import test_utils  # noqa: F401

import axon
from test_utils import create_server


def test_simplified_register_multiple_params():
    """Test simplified register_function_raw with multiple parameters."""

    async def add_three(a: int, b: int, c: int) -> int:
        return a + b + c

    with create_server("test_worker") as runtime:
        # Use simplified API - types are auto-inferred
        runtime.register_function_raw(
            function_id=1,
            callable=add_three,
        )


def test_simplified_register_multiple_returns():
    """Test simplified register_function_raw with multiple return values."""

    async def multi_return(x: int, y: int) -> Tuple[int, int]:
        return (x + y, x * y)

    with create_server("test_worker") as runtime:
        # Use simplified API - return types are auto-inferred from Tuple[int, int]
        runtime.register_function_raw(
            function_id=2,
            callable=multi_return,
        )


def test_simplified_register_mixed_params_and_returns():
    """Test simplified register_function_raw with mixed params and multiple returns."""

    async def complex_func(x: int, name: str, f: float) -> Tuple[int, str, float]:
        return (x * 2, f"Hello {name}", f + 1.0)

    with create_server("test_worker") as runtime:
        # Use simplified API
        runtime.register_function_raw(
            function_id=3,
            callable=complex_func,
        )


def test_simplified_register_list_params():
    """Test simplified register_function_raw with List parameter types."""

    async def process_list(numbers: List[int]) -> List[int]:
        return [x * 2 for x in numbers]

    with create_server("test_worker") as runtime:
        # Use simplified API
        runtime.register_function_raw(
            function_id=4,
            callable=process_list,
        )


def test_simplified_register_tuple_return_with_list():
    """Test simplified register_function_raw with Tuple return containing List."""

    async def process_and_count(
        numbers: List[int],
    ) -> Tuple[int, List[int]]:
        return (len(numbers), [x * 2 for x in numbers])

    with create_server("test_worker") as runtime:
        # Use simplified API
        runtime.register_function_raw(
            function_id=5,
            callable=process_and_count,
        )


def test_simplified_register_no_annotations():
    """Test simplified register_function_raw with no type annotations (should use UNKNOWN)."""

    async def untyped_func(x, y):
        return x + y

    with create_server("test_worker") as runtime:
        # Use simplified API - should fallback to UNKNOWN types
        runtime.register_function_raw(
            function_id=6,
            callable=untyped_func,
        )


def test_simplified_register_void_return():
    """Test simplified register_function_raw with void return type."""

    async def void_func(x: int) -> None:
        pass

    with create_server("test_worker") as runtime:
        # Use simplified API
        runtime.register_function_raw(
            function_id=7,
            callable=void_func,
        )


def test_simplified_register_custom_function_name():
    """Test simplified register_function_raw with custom function name."""

    async def my_func(x: int) -> int:
        return x * 2

    with create_server("test_worker") as runtime:
        # Use simplified API with custom function name
        runtime.register_function_raw(
            function_id=8,
            callable=my_func,
            function_name="custom_name",
        )


def test_simplified_register_with_policies():
    """Test simplified register_function_raw with memory and lifecycle policies."""

    async def func_with_policies(x: int) -> int:
        return x * 2

    with create_server("test_worker") as runtime:
        # Use simplified API with policies
        runtime.register_function_raw(
            function_id=9,
            callable=func_with_policies,
            memory_policy=None,
            lifecycle_policy=None,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
