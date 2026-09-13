"""Score and sample optimizer candidates using their Pareto-frontier coverage."""

# https://github.com/gepa-ai/gepa

import logging
import random
from collections.abc import Mapping, Sequence
from typing import TypeVar

from .serialization import fields

Key = TypeVar("Key")
_logger = logging.getLogger(__name__)


def json_default(x: object) -> object:
    """Describe unsupported output values for diagnostic JSON exports.

    Returns
    -------
    object
        Mapping fields when available, or a printable representation of the value.

    """
    try:
        if isinstance(x, Mapping):
            return dict(fields(x))
        return repr(x)
    except Exception:
        _logger.debug("Could not extract output mapping fields", exc_info=True)
        return repr(x)


def idxmax(lst: list[float]) -> int:
    """Find the first index attaining the maximum score.

    Returns
    -------
    int
        The first maximal score index.

    """
    max_val = max(lst)
    return lst.index(max_val)


def is_dominated(
    y: int,
    programs: set[int],
    program_at_pareto_front_valset: Mapping[Key, set[int]],
) -> bool:
    """Check whether other candidates cover every frontier containing a candidate.

    Returns
    -------
    bool
        Whether all of the candidate's frontier coverage is redundant.

    """
    y_fronts = [
        front for front in program_at_pareto_front_valset.values() if y in front
    ]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False

    return True


def _dominated_indices(
    programs: list[int],
    frontiers: Mapping[Key, set[int]],
) -> set[int]:
    dominated: set[int] = set()
    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(
                y,
                set(programs).difference({y}).difference(dominated),
                frontiers,
            ):
                dominated.add(y)
                found_to_remove = True
                break

    return dominated


def remove_dominated_programs(
    program_at_pareto_front_valset: Mapping[Key, set[int]],
    scores: Sequence[float] | Mapping[int, float] | None = None,
) -> dict[Key, set[int]]:
    """Remove candidates whose frontier coverage is provided by better candidates.

    Returns
    -------
    dict
        The original frontier keys with redundant candidate indices removed.

    Raises
    ------
    RuntimeError
        If the remaining candidates fail to cover a nonempty frontier.

    """
    freq: dict[int, int] = {}
    for front in program_at_pareto_front_valset.values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    programs = list(freq.keys())

    if scores is None:
        scores = dict.fromkeys(programs, 1)

    def score_for_program(program: int) -> float:
        return scores[program]

    programs = sorted(programs, key=score_for_program, reverse=False)

    dominated = _dominated_indices(programs, program_at_pareto_front_valset)

    dominators = [p for p in programs if p not in dominated]
    for front in program_at_pareto_front_valset.values():
        if not front:
            continue
        if not (any(p in front for p in dominators)):
            message = (
                "Invalid optimization state: any((p in front for p in dominators))"
            )
            raise RuntimeError(message)

    new_program_at_pareto_front_valset = {
        val_id: {prog_idx for prog_idx in front if prog_idx in dominators}
        for val_id, front in program_at_pareto_front_valset.items()
    }
    for val_id, front_new in new_program_at_pareto_front_valset.items():
        if not (front_new.issubset(program_at_pareto_front_valset[val_id])):
            message = (
                "Invalid optimization state: front_new.issubset(pro"
                "gram_at_pareto_front_valset[val_id])"
            )
            raise RuntimeError(message)

    return new_program_at_pareto_front_valset


def find_dominator_programs(
    pareto_front_programs: Mapping[Key, set[int]],
    train_val_weighted_agg_scores_for_all_programs: list[float],
) -> list[int]:
    """Collect candidates retaining nonredundant Pareto-frontier coverage.

    Returns
    -------
    list[int]
        The unique candidate indices that remain after removing dominated entries.

    """
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(
        train_val_pareto_front_programs,
        scores=train_val_weighted_agg_scores_for_all_programs,
    )
    uniq_progs: list[int] = []
    for front in new_program_at_pareto_front_valset.values():
        uniq_progs.extend(front)
    return list(set(uniq_progs))


def select_program_candidate_from_pareto_front(
    pareto_front_programs: Mapping[Key, set[int]],
    train_val_weighted_agg_scores_for_all_programs: list[float],
    rng: random.Random,
) -> int:
    """Sample a nondominated candidate weighted by its frontier coverage.

    Returns
    -------
    int
        The sampled candidate index.

    Raises
    ------
    RuntimeError
        If no candidate remains available for sampling.

    """
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(
        train_val_pareto_front_programs,
        scores=train_val_weighted_agg_scores_for_all_programs,
    )
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset.values():
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1

    sampling_list = [
        prog_idx
        for prog_idx, freq in program_frequency_in_validation_pareto_front.items()
        for _ in range(freq)
    ]

    if not (len(sampling_list) > 0):
        message = "Invalid optimization state: len(sampling_list) > 0"
        raise RuntimeError(message)

    return rng.choice(sampling_list)
