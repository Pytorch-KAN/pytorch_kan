import rich.console
import pydantic.deprecated.class_validators as class_validators

import rich.console
import pydantic.deprecated.class_validators as class_validators


# Patch Pydantic's root_validator to default skip_on_failure=True for v2 compatibility
_orig_root_validator = class_validators.root_validator

def _patched_root_validator(*args, **kwargs):  # pragma: no cover - tiny helper
    kwargs.setdefault("skip_on_failure", True)
    return _orig_root_validator(*args, **kwargs)

class_validators.root_validator = _patched_root_validator


class _LoguruCompatibleConsole(rich.console.Console):
    """Rich console with a write method for Loguru compatibility."""

    def write(self, message: str) -> None:  # pragma: no cover - trivial
        # Loguru expects sinks to implement write(). Delegate to print().
        self.print(message, end="")


rich.console.Console = _LoguruCompatibleConsole
