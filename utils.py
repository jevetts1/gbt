import math

def progress_bar(progress):
    bar_filled = "=" * math.ceil(progress * 40)
    bar_unfilled = ":" * math.floor((1 - progress) * 40)

    bar = "[" + bar_filled + ">" + bar_unfilled + "]"

    return bar