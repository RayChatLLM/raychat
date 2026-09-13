"""Combine compatible candidate descendants using shared-ancestor evidence."""

# https://github.com/gepa-ai/gepa

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic

from .adapter import Candidate, DataInst, RolloutOutput
from .data_loader import DataId, DataLoader
from .gepa_utils import find_dominator_programs
from .logger import LoggerProtocol
from .proposer import CandidateProposal, ProposeNewCandidate
from .state import GEPAState, ObjectiveScores, ProgramIdx
from .type_support import override

AncestorLog = tuple[int, int, int]
MergeDescription = tuple[int, int, tuple[int, ...]]
MergeAttempt = tuple[Candidate, ProgramIdx, ProgramIdx, ProgramIdx] | None


_PARENT_COUNT = 2
_MINIMUM_MERGE_HISTORY = 3


@dataclass(frozen=True, kw_only=True)
class MergeHistory:
    """Candidate ancestry, scores and previous combinations used to plan merges."""

    candidates: Sequence[Candidate]
    parents: Sequence[Sequence[int | None]]
    scores: Sequence[float]
    performed: tuple[list[AncestorLog], list[MergeDescription]]


@dataclass(frozen=True, kw_only=True)
class MergeSettings:
    """Merge attempt limits, required validation overlap and the shared generator."""

    max_invocations: int
    val_overlap_floor: int = 5
    rng: random.Random | None = None


def does_triplet_have_desirable_predictors(
    program_candidates: Sequence[Candidate],
    ancestor: ProgramIdx,
    id1: ProgramIdx,
    id2: ProgramIdx,
) -> bool:
    """Check whether descendants improved different components of their ancestor.

    Returns
    -------
    bool
        Whether at least one component can inherit a distinct descendant's change.

    """
    found_predictors: list[tuple[int, int]] = []
    pred_names = list(program_candidates[ancestor].keys())
    for pred_idx, pred_name in enumerate(pred_names):
        pred_anc = program_candidates[ancestor][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]
        if pred_anc in {pred_id1, pred_id2} and pred_id1 != pred_id2:
            same_as_ancestor_id = 1 if pred_anc == pred_id1 else 2
            found_predictors.append((pred_idx, same_as_ancestor_id))
    return bool(found_predictors)


def filter_ancestors(
    i: ProgramIdx,
    j: ProgramIdx,
    common_ancestors: Iterable[ProgramIdx],
    history: MergeHistory,
) -> list[ProgramIdx]:
    """Keep unused ancestors with compatible components and no higher score.

    Returns
    -------
    list[ProgramIdx]
        Compatible ancestor indices in their original iteration order.

    """
    filtered: list[ProgramIdx] = []
    for ancestor in common_ancestors:
        if (i, j, ancestor) in history.performed[0]:
            continue
        if (
            history.scores[ancestor] > history.scores[i]
            or history.scores[ancestor] > history.scores[j]
        ):
            continue
        if does_triplet_have_desirable_predictors(history.candidates, ancestor, i, j):
            filtered.append(ancestor)
    return filtered


def find_common_ancestor_pair(
    history: MergeHistory,
    rng: random.Random,
    program_indexes: Sequence[int],
    max_attempts: int = 10,
) -> tuple[int, int, int] | None:
    """Sample two independent descendants and a compatible shared ancestor.

    Returns
    -------
    tuple[int, int, int] | None
        Descendant indices and their ancestor, or None after exhausting attempts.

    """

    def get_ancestors(node: int, ancestors_found: set[int]) -> list[int]:
        for parent in history.parents[node]:
            if parent is not None and parent not in ancestors_found:
                ancestors_found.add(parent)
                get_ancestors(parent, ancestors_found)
        return list(ancestors_found)

    for _ in range(max_attempts):
        if len(program_indexes) < _PARENT_COUNT:
            return None
        i, j = rng.sample(list(program_indexes), _PARENT_COUNT)
        if i == j:
            continue
        if j < i:
            i, j = j, i
        ancestors_i = get_ancestors(i, set())
        ancestors_j = get_ancestors(j, set())
        if j in ancestors_i or i in ancestors_j:
            continue
        common = filter_ancestors(i, j, set(ancestors_i) & set(ancestors_j), history)
        if common:
            ancestor = rng.choices(
                common,
                k=1,
                weights=[history.scores[a] for a in common],
            )[0]
            return i, j, ancestor
    return None


def _choose_predictor_source(
    history: MergeHistory,
    rng: random.Random,
    triplet: AncestorLog,
    name: str,
) -> ProgramIdx:
    id1, id2, ancestor = triplet
    original = history.candidates[ancestor][name]
    left, right = history.candidates[id1][name], history.candidates[id2][name]
    if original in {left, right} and left != right:
        return id2 if original == left else id1
    if original not in {left, right}:
        if history.scores[id1] > history.scores[id2]:
            return id1
        if history.scores[id2] > history.scores[id1]:
            return id2
        return rng.choice([id1, id2])
    return id1


def _merge_predictors(
    history: MergeHistory,
    rng: random.Random,
    triplet: AncestorLog,
) -> tuple[Candidate, tuple[ProgramIdx, ...]]:
    id1, id2, ancestor = triplet
    if (
        not history.scores[ancestor] <= history.scores[id1]
        or not history.scores[ancestor] <= history.scores[id2]
    ):
        message = "Ancestor should not be better than its descendants"
        raise RuntimeError(message)
    if id1 == id2:
        message = "Cannot merge the same program"
        raise RuntimeError(message)
    names = set(history.candidates[ancestor])
    if names != set(history.candidates[id1]) or names != set(history.candidates[id2]):
        message = "Predictors should be the same across all programs"
        raise RuntimeError(message)
    candidate = dict(history.candidates[ancestor])
    sources: tuple[ProgramIdx, ...] = ()
    for name in names:
        source = _choose_predictor_source(history, rng, triplet, name)
        candidate[name] = history.candidates[source][name]
        sources = (*sources, source)
    return candidate, sources


def sample_and_attempt_merge_programs_by_common_predictors(
    history: MergeHistory,
    rng: random.Random,
    merge_candidates: Sequence[int],
    has_val_support_overlap: Callable[[ProgramIdx, ProgramIdx], bool] | None = None,
    max_attempts: int = 10,
) -> MergeAttempt:
    """Combine complementary descendant components without repeating prior merges.

    Returns
    -------
    MergeAttempt
        The merged candidate, its parents and ancestor, or None when unavailable.

    """
    if (
        len(merge_candidates) < _PARENT_COUNT
        or len(history.parents) < _MINIMUM_MERGE_HISTORY
    ):
        return None
    for _ in range(max_attempts):
        triplet = find_common_ancestor_pair(
            history,
            rng,
            list(merge_candidates),
            max_attempts,
        )
        if triplet is None or triplet in history.performed[0]:
            continue
        id1, id2, ancestor = triplet
        candidate, sources = _merge_predictors(history, rng, triplet)
        if (id1, id2, sources) in history.performed[1]:
            continue
        if has_val_support_overlap and not has_val_support_overlap(id1, id2):
            continue
        history.performed[1].append((id1, id2, sources))
        return candidate, id1, id2, ancestor
    return None


class MergeProposer(
    ProposeNewCandidate[DataId, RolloutOutput],
    Generic[DataId, DataInst, RolloutOutput],
):
    """Implements merge flow that combines compatible descendants of a common ancestor.

    - Find merge candidates among Pareto front dominators
    - Attempt a merge via sample_and_attempt_merge_programs_by_common_predictors
    - Subsample eval on valset-driven selected indices
    - Return proposal if merge's subsample score >= max(parents)
    The engine handles full eval + adding to state.
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        valset: DataLoader[DataId, DataInst],
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
        settings: MergeSettings,
    ) -> None:
        """Bind merge evaluation to explicit scheduling and overlap settings.

        Raises
        ------
        ValueError
            Required validation overlap is not positive.

        """
        self.logger = logger
        self.valset = valset
        self.evaluator = evaluator
        self.max_merge_invocations = settings.max_invocations
        self.rng = settings.rng if settings.rng is not None else random.Random(0)
        if settings.val_overlap_floor <= 0:
            message = "val_overlap_floor should be a positive integer"
            raise ValueError(message)
        self.val_overlap_floor = settings.val_overlap_floor
        # Internal counters matching original behavior
        self.merges_due = 0
        self.total_merges_tested = 0
        self.merges_performed: tuple[list[AncestorLog], list[MergeDescription]] = (
            [],
            [],
        )

        # Toggle controlled by engine: set True when last iter found new program
        self.last_iter_found_new_program = False

    def select_eval_subsample_for_merged_program(
        self,
        scores1: dict[DataId, float],
        scores2: dict[DataId, float],
        num_subsample_ids: int = 5,
    ) -> list[DataId]:
        """Sample overlapping validation examples favoring each parent's strengths.

        Returns
        -------
        list[DataId]
            A balanced sample of identifiers with replacement when coverage is small.

        """
        common_ids = list(set(scores1.keys()) & set(scores2.keys()))

        p1 = [idx for idx in common_ids if scores1[idx] > scores2[idx]]
        p2 = [idx for idx in common_ids if scores2[idx] > scores1[idx]]
        p3 = [idx for idx in common_ids if idx not in p1 and idx not in p2]

        n_each = max(1, math.ceil(num_subsample_ids / 3))
        selected: list[DataId] = []
        for bucket in (p1, p2, p3):
            if len(selected) >= num_subsample_ids:
                break
            available = [idx for idx in bucket if idx not in selected]
            take = min(len(available), n_each, num_subsample_ids - len(selected))
            if take > 0:
                selected += self.rng.sample(available, k=take)

        remaining = num_subsample_ids - len(selected)
        if remaining > 0:
            unused = [idx for idx in common_ids if idx not in selected]
            if len(unused) >= remaining:
                selected += self.rng.sample(unused, k=remaining)
            elif common_ids:
                selected += self.rng.choices(common_ids, k=remaining)

        return selected[:num_subsample_ids]

    @override
    def propose(
        self,
        state: GEPAState[RolloutOutput, DataId],
    ) -> CandidateProposal[DataId] | None:
        """Evaluate a scheduled merge on overlapping validation support.

        Returns
        -------
        CandidateProposal | None
            A candidate with comparison scores, or None when no merge is available.

        Raises
        ------
        RuntimeError
            A sampled identifier lacks a parent's required validation result.

        """
        i = state.i + 1
        state.full_program_trace[-1]["invoked_merge"] = True

        # Only attempt when scheduled by engine and after a new program in last
        # iteration
        if not (self.last_iter_found_new_program and self.merges_due > 0):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        merge_output = self._find_merge(state)

        if merge_output is None:
            self.logger.log(f"Iteration {i}: No merge candidates found")
            return None

        new_program, id1, id2, ancestor = merge_output
        state.full_program_trace[-1]["merged"] = True
        state.full_program_trace[-1]["merged_entities"] = (id1, id2, ancestor)
        self.merges_performed[0].append((id1, id2, ancestor))
        self.logger.log(
            f"Iteration {i}: Merged programs {id1} and {id2} via ancestor {ancestor}",
        )

        subsample_ids = self.select_eval_subsample_for_merged_program(
            state.prog_candidate_val_subscores[id1],
            state.prog_candidate_val_subscores[id2],
        )
        if not subsample_ids:
            self.logger.log(
                (
                    "Iteration "
                    f"{i}"
                    ": Skipping merge of "
                    f"{id1}"
                    " and "
                    f"{id2}"
                    " due to insufficient overlapping val coverage"
                ),
            )
            return None

        if not (
            set(subsample_ids).issubset(state.prog_candidate_val_subscores[id1].keys())
        ):
            message = (
                "Invalid optimization state: set(subsample_ids).iss"
                "ubset(state.prog_candidate_val_subscores[id1].keys"
                "())"
            )
            raise RuntimeError(message)
        if not (
            set(subsample_ids).issubset(state.prog_candidate_val_subscores[id2].keys())
        ):
            message = (
                "Invalid optimization state: set(subsample_ids).iss"
                "ubset(state.prog_candidate_val_subscores[id2].keys"
                "())"
            )
            raise RuntimeError(message)
        id1_sub_scores = [
            state.prog_candidate_val_subscores[id1][k] for k in subsample_ids
        ]
        id2_sub_scores = [
            state.prog_candidate_val_subscores[id2][k] for k in subsample_ids
        ]
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids

        _outputs_by_id, scores_by_id, _objective_by_id, actual_evals_count = (
            state.cached_evaluate_full(
                new_program,
                subsample_ids,
                self.valset.fetch,
                self.evaluator,
            )
        )
        new_sub_scores = [scores_by_id[eid] for eid in subsample_ids]

        state.full_program_trace[-1]["id1_subsample_scores"] = id1_sub_scores
        state.full_program_trace[-1]["id2_subsample_scores"] = id2_sub_scores
        state.full_program_trace[-1]["new_program_subsample_scores"] = new_sub_scores

        # Account for actual evaluations, excluding cache hits.
        state.increment_evals(actual_evals_count)

        # Acceptance will be evaluated by engine (>= max(parents))
        return CandidateProposal(
            candidate=new_program,
            parent_program_ids=[id1, id2],
            subsample_indices=subsample_ids,
            subsample_scores_before=[sum(id1_sub_scores), sum(id2_sub_scores)],
            subsample_scores_after=new_sub_scores,
            tag="merge",
        )

    def _find_merge(self, state: GEPAState[RolloutOutput, DataId]) -> MergeAttempt:
        tracked_scores = state.program_full_scores_val_set
        candidates = find_dominator_programs(
            state.get_pareto_front_mapping(),
            list(tracked_scores),
        )

        def has_overlap(id1: ProgramIdx, id2: ProgramIdx) -> bool:
            common = set(state.prog_candidate_val_subscores[id1]) & set(
                state.prog_candidate_val_subscores[id2],
            )
            return len(common) >= self.val_overlap_floor

        return sample_and_attempt_merge_programs_by_common_predictors(
            history=MergeHistory(
                candidates=state.program_candidates,
                parents=state.parent_program_for_candidate,
                scores=list(tracked_scores),
                performed=self.merges_performed,
            ),
            rng=self.rng,
            merge_candidates=candidates,
            has_val_support_overlap=has_overlap,
        )
