def mask_input(x, mask):
    if not mask == None:
        y = x * mask
    else:
        y = x
    return y
