import pytest
from typing import Tuple, List, Dict, Any, Union
import axon
import test_utils
from test_utils import create_server


# Define some helper types/functions for testing
class CustomClass:
    pass


def test_register_function_valid_types():
    """Test register_function with valid supported types."""

    async def valid_func(x: int, y: float, name: str, active: bool) -> str:
        return f"{name}: {x + y} is {active}"

    with create_server("test_worker_valid") as runtime:
        runtime.register_function(
            callable=valid_func,
            function_id=1,
        )


def test_register_function_valid_list():
    """Test register_function with valid List types."""

    async def list_func(numbers: List[int]) -> List[float]:
        return [float(x) for x in numbers]

    with create_server("test_worker_list") as runtime:
        runtime.register_function(
            callable=list_func,
            function_id=2,
        )


def test_register_function_valid_list_str():
    """Test register_function with valid List[str] types."""

    async def list_str_func(names: List[str]) -> List[str]:
        return [f"Hello {name}" for name in names]

    with create_server("test_worker_list_str") as runtime:
        runtime.register_function(
            callable=list_str_func,
            function_id=201,
        )


def test_register_function_valid_nested_list():
    """Test register_function with valid nested List types."""

    async def nested_list_func(matrix: List[List[int]]) -> List[List[float]]:
        return [[float(x) for x in row] for row in matrix]

    with create_server("test_worker_nested_list") as runtime:
        runtime.register_function(
            callable=nested_list_func,
            function_id=202,
        )


def test_register_function_valid_tuple_return():
    """Test register_function with valid Tuple return."""

    async def tuple_func(x: int) -> Tuple[int, str]:
        return (x, str(x))

    with create_server("test_worker_tuple") as runtime:
        runtime.register_function(
            callable=tuple_func,
            function_id=3,
        )


def test_register_function_void_return():
    """Test register_function with None return type (void)."""

    async def void_func(x: int) -> None:
        pass

    with create_server("test_worker_void") as runtime:
        runtime.register_function(
            callable=void_func,
            function_id=4,
        )


def test_register_function_invalid_unknown_param():
    """Test register_function errors on unknown parameter type."""

    async def invalid_param(x: CustomClass) -> int:
        return 1

    with create_server("test_worker_invalid") as runtime:
        with pytest.raises(
            TypeError, match="Unsupported or missing type annotation for parameter 0"
        ):
            runtime.register_function(
                callable=invalid_param,
                function_id=5,
            )


def test_register_function_invalid_unknown_return():
    """Test register_function errors on unknown return type."""

    async def invalid_return(x: int) -> CustomClass:
        return CustomClass()

    with create_server("test_worker_invalid_ret") as runtime:
        with pytest.raises(
            TypeError, match="Unsupported or missing return type annotation"
        ):
            runtime.register_function(
                callable=invalid_return,
                function_id=6,
            )


def test_register_function_invalid_untyped_list():
    """Test register_function errors on untyped List."""

    async def invalid_list(x: List) -> int:
        return 1

    with create_server("test_worker_invalid_list") as runtime:
        with pytest.raises(
            TypeError, match="List without type argument is not supported"
        ):
            runtime.register_function(
                callable=invalid_list,
                function_id=7,
            )


def test_register_function_invalid_dict():
    """Test register_function errors on Dict type (unsupported)."""

    async def invalid_dict(x: Dict[str, int]) -> int:
        return 1

    with create_server("test_worker_invalid_dict") as runtime:
        with pytest.raises(TypeError, match="Unsupported or missing type annotation"):
            runtime.register_function(
                callable=invalid_dict,
                function_id=8,
            )


def test_register_function_invalid_union():
    """Test register_function errors on Union type (unsupported)."""

    async def invalid_union(x: Union[int, float]) -> int:
        return 1

    with create_server("test_worker_invalid_union") as runtime:
        with pytest.raises(TypeError, match="Unsupported or missing type annotation"):
            runtime.register_function(
                callable=invalid_union,
                function_id=9,
            )


def test_register_function_missing_annotation():
    """Test register_function errors on missing annotation."""

    async def missing_anno(x, y: int) -> int:
        return x + y

    with create_server("test_worker_missing") as runtime:
        with pytest.raises(TypeError, match="Unsupported or missing type annotation"):
            runtime.register_function(
                callable=missing_anno,
                function_id=10,
            )


def test_register_function_dlpack():
    """Test register_function with object having __dlpack__ method."""

    class MockTensor:
        def __dlpack__(self, stream=None):
            pass

    async def tensor_func(x: MockTensor) -> MockTensor:
        return x

    with create_server("test_worker_dlpack") as runtime:
        runtime.register_function(
            callable=tensor_func,
            function_id=11,
        )


def test_register_function_invalid_list_list_str():
    """Test register_function errors on List[List[str]] (unsupported nested string list)."""

    async def nested_str_func(matrix: List[List[str]]) -> int:
        return 1

    with create_server("test_worker_nested_str") as runtime:
        with pytest.raises(TypeError, match="Unsupported inner element type"):
            runtime.register_function(
                callable=nested_str_func,
                function_id=203,
            )


def test_register_function_invalid_tuple_param():
    """Test register_function errors on Tuple as a parameter type (unsupported)."""

    async def tuple_param_func(x: Tuple[int, str]) -> int:
        return 1

    with create_server("test_worker_tuple_param") as runtime:
        with pytest.raises(TypeError, match="Unsupported or missing type annotation"):
            runtime.register_function(
                callable=tuple_param_func,
                function_id=204,
            )


def test_register_function_invalid_list_list_untyped():
    """Test register_function errors on List[List] without inner type args."""

    async def untyped_nested(matrix: List[List]) -> int:
        return 1

    with create_server("test_worker_nested_untyped") as runtime:
        with pytest.raises(
            TypeError, match="requires explicit element type annotation"
        ):
            runtime.register_function(
                callable=untyped_nested,
                function_id=205,
            )


def test_register_function_async_callable():
    """Test register_function with a callable class instance."""

    class AsyncCallable:
        async def __call__(self, x: int) -> int:
            return x

    with create_server("test_worker_callable") as runtime:
        # Should not raise any TypeError regarding IsAsyncFunction
        runtime.register_function(
            callable=AsyncCallable(),
            function_id=206,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
