from typing import Any, Callable


def tool(*args, **kwargs):  # type: ignore
    """Lightweight tool decorator compatible with various call styles.

    Accepts either: @tool("name", return_direct=False) or @tool(name="name", ...)
    Always returns a wrapper with .name and .invoke(kwargs).
    """

    # Extract name and return_direct from positional/keyword args
    name: str = ""
    if args and isinstance(args[0], str):
        name = args[0]
    else:
        name = kwargs.get("name", "")
    return_direct: bool = bool(kwargs.get("return_direct", False))

    def decorator(func: Callable[..., Any]):
        class _DummyTool:
            def __init__(self, f: Callable[..., Any]):
                self.name = name or f.__name__
                self.description = (f.__doc__ or "").strip()
                self._func = f

            def invoke(self, input: Any) -> Any:
                if isinstance(input, dict):
                    return self._func(**input)
                return self._func(input)

            def __call__(self, *cargs: Any, **ckwargs: Any) -> Any:
                return self._func(*cargs, **ckwargs)

        return _DummyTool(func)

    return decorator

