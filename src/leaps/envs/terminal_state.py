"""Terminal-state  conditions for muscle-actuated locomotion.

Gaitgym get_done() method
"""

# Same candidate list as sconegym.gaitgym.SconeGym._find_head_body(). The
# search doesn't stop at the first match -- it keeps the *last* body in
# model.bodies() whose name is in this list. H0918 only has "torso", but
# H1622 and H2190 have both "torso" and "head", so on those two models this
# ends up picking "head" (it's defined after "torso"). Replicated as-is.
HEAD_BODY_NAMES = ("torso", "head", "lumbar")


def find_head_body(model):
    """Return the head/torso body sconegym uses for its height check."""
    head_body = None
    for b in model.bodies():
        if b.name() in HEAD_BODY_NAMES:
            head_body = b
    if head_body is None:
        raise ValueError(f"No body named any of {HEAD_BODY_NAMES} in model")
    return head_body


class NoTerminal:
    """Never ends an episode early -- fixed-length rollouts only."""

    def reset(self) -> None:
        pass

    def __call__(self, model) -> bool:
        return False


class HeightTerminal:
    """Falls when COM or head/torso height drops below a threshold.

    Same condition as sconegym.gaitgym.GaitGym._get_done(): com_pos().y <
    min_com_height, or the head/torso body's com_pos().y < min_head_height.
    """

    def __init__(
        self,
        min_com_height: float = 0.5,
        min_head_height: float = 0.9,
        fall_recovery_time: float = 0.0,
    ):
        self.min_com_height = min_com_height
        self.min_head_height = min_head_height
        self.fall_recovery_time = fall_recovery_time
        self._head_body = None
        self._fall_time = -1.0

    def reset(self) -> None:
        """Call on episode reset -- clears the cached head body and fall timer."""
        self._head_body = None
        self._fall_time = -1.0

    def __call__(self, model) -> bool:
        if self._head_body is None:
            self._head_body = find_head_body(model)

        fall = model.com_pos().y < self.min_com_height
        fall = fall or self._head_body.com_pos().y < self.min_head_height

        t = model.time()
        if fall:
            if self._fall_time < 0:
                self._fall_time = t
            return (t - self._fall_time) >= self.fall_recovery_time
        self._fall_time = -1.0
        return False
