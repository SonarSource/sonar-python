from typing import Any, ClassVar, Callable

class View:
    methods: ClassVar[list[str] | None]
    provide_automatic_options: ClassVar[bool | None]
    decorators: ClassVar[list[Callable[..., Any]]]

    def dispatch_request(self) -> Any: ...

    @classmethod
    def as_view(cls, name: str, *class_args: Any, **class_kwargs: Any) -> Callable[..., Any]: ...

