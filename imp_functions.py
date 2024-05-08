# Function to compute Euclidean Distance between two points
def euclidean_distance(ptA, ptB):
    dist = ((ptA[0] - ptB[0]) ** 2) + ((ptA[1] - ptB[1]) ** 2)
    dist = dist ** 0.5

    return dist

# Function to detect eye blinks
def isBlinked(a, b, c, d, e, f):
    vertical1 = euclidean_distance(b, d)
    vertical2 = euclidean_distance(c, e)
    horizontal = euclidean_distance(a, f)

    ratio = (vertical1 + vertical2) / horizontal
    if ratio < 0.5:
        return True
    else:
        return False

# Function to check if driver is yawning
def isYawned(a, b, c, d, e, f, g, h):
    vertical1 = euclidean_distance(b, h)
    vertical2 = euclidean_distance(c, g)
    vertical3 = euclidean_distance(d, f)
    horizontal = euclidean_distance(a, e)

    ratio = (vertical1 + vertical2 + vertical3) / horizontal
    if ratio >= 0.75:
        return True
    else:
        return False
