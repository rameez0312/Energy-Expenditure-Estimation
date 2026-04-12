import pandas as pd


def generate_window_starts(start_time, end_time, window_size=30, step_size=15):
    """
    Generate consistent window start timestamps.
    """

    starts = []
    current = start_time

    while current + pd.Timedelta(seconds=window_size) <= end_time:
        starts.append(current)
        current += pd.Timedelta(seconds=step_size)

    return starts