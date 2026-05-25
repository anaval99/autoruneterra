MODES = ["confirmation", "prepare_summon", "prepare_battle", "unknown"]

# Diamond-shaped hitbox around the confirmation button, normalized 0..100.
CONFIRMATION_HITBOX = [
    (84, 50),
    (87, 46),
    (90, 51),
    (87, 54),
]


def _point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def coord_to_mode(x, y):
    """Map a normalized click coordinate (0..100) to a mode label."""
    if _point_in_polygon(x, y, CONFIRMATION_HITBOX):
        return "confirmation"
    if y > 91:
        return "prepare_summon"
    if 78 <= y <= 88:
        return "prepare_battle"
    return "unknown"
