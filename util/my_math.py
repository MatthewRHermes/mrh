
def round_thresh (the_float, the_thresh):
    round_diff = round (the_float) - the_float
    round_sign = -1 if round_diff<0 else 1
    round_diff = round_sign * (min (abs (round_diff), abs (the_thresh)))
    return the_float + round_diff
    
def is_close_to_integer (the_float, the_thresh):
    return round_thresh (the_float, the_thresh).is_integer ()

