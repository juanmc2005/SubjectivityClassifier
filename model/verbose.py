from tqdm import tqdm


def vlist(xs, desc, verbose):
    return tqdm(xs, desc=desc) if verbose else xs


def vprint(desc, verbose):
    if verbose:
        print(desc)
