def callback_fn(curr_value: float | int, best_value: float | int, patience_counter: int, patience: int) -> tuple[bool, int]:
    """
    Implements early stopping logic.
    Returns a tuple: (should_stop, updated_patience_counter).
    """
    if curr_value < best_value:
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            return True, patience_counter
    return False, patience_counter