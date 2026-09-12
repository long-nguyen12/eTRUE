"""Consistent error capture for optional external operations."""


def optional_result(errors, label, operation):
    """Return an optional result while recording a non-fatal failure."""
    try:
        return operation()
    except Exception as error:
        errors.append(label + ": " + str(error))
        return None
