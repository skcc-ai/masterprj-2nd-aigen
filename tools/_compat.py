from typing import Any, Callable


try:
    from langchain.tools import tool as _lc_tool  # type: ignore

    def tool(name: str, return_direct: bool = False):
        return _lc_tool(name=name, return_direct=return_direct)

except Exception:

    def tool(name: str, return_direct: bool = False):  # type: ignore
        """Lightweight fallback decorator when LangChain is unavailable.

        Wraps a function into an object with `.name` and `.invoke(kwargs)` for demo/testing.
        """

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

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    return self._func(*args, **kwargs)

            return _DummyTool(func)

        return decorator


