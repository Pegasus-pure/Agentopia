"""Social Ranking and Reward calculation.

This module implements the Reward system with three components:
1. Social Reward: Social standing via PageRank on affection/respect graphs
2. Subjective Reward: Personal fulfillment via data-driven evaluation
3. Economy Reward: Economic change (deposit delta) over the reward period

Additionally provides:
- SocialMetrics: Absolute social metrics for cross-run comparison
- TotalReward: Combined social + subjective + economy reward
- Return calculation: Discounted future rewards for RL optimization
- Advantage calculation: Return_{t+1} - Return_t for policy gradient
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.config import get_config
from src.utils import extract_json
from src.world.clock import TimeState

if TYPE_CHECKING:
    from src.agents.role_agent import RoleAgent

# Type alias for social graph: Dict[from_agent, Dict[to_agent, weight]]
SocialGraph = Dict[str, Dict[str, float]]

# Neutral baseline score for social evaluation (60 = threshold for positive)
SOCIAL_NEUTRAL_SCORE = 60


def _derive_ranking(scores: Dict[str, int]) -> List[str]:
    """Derive ranking list from score dict (desc by score, name tie-breaker)."""
    return sorted(scores.keys(), key=lambda n: (-scores[n], n))


@dataclass
class SocialRanking:
    """Single Agent's social evaluation of other people (score-based)."""

    agent_name: str
    time: str  # TimeState.__str__() format
    affection_scores: Dict[str, int]  # name → 0-100 score
    respect_scores: Dict[str, int]  # name → 0-100 score

    @property
    def affection_ranking(self) -> List[str]:
        return _derive_ranking(self.affection_scores)

    @property
    def respect_ranking(self) -> List[str]:
        return _derive_ranking(self.respect_scores)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SocialRanking":
        return SocialRanking(
            agent_name=d["agent_name"],
            time=d["time"],
            affection_scores=d["affection_scores"],
            respect_scores=d["respect_scores"],
        )


@dataclass
class SocialReward:
    """Single Agent's social achievement (from others' evaluation)."""

    agent_name: str
    time: str
    affection_score: float  # PageRank score (0-1)
    respect_score: float  # PageRank score (0-1)
    combined_score: float  # Weighted combination

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SocialReward":
        return SocialReward(
            agent_name=d["agent_name"],
            time=d["time"],
            affection_score=d["affection_score"],
            respect_score=d["respect_score"],
            combined_score=d["combined_score"],
        )


@dataclass
class SubjectiveReward:
    """Single Agent's subjective fulfillment reward.

    Calculated from fulfillment history: mean of all values with misery penalty applied.
    """

    agent_name: str
    time: str
    score: float  # Mean after penalty applied to values below threshold
    n_penalties: int  # Number of (week, dim) below threshold

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SubjectiveReward":
        return SubjectiveReward(
            agent_name=d["agent_name"],
            time=d["time"],
            score=d["score"],
            n_penalties=d["n_penalties"],
        )


@dataclass
class TotalReward:
    """Combined reward for a single Agent at a settlement point.

    Total = social_weight * social_z + subjective_weight * subj_z + economy_weight * econ_z
    All components are z-score normalized before combining.
    """

    agent_name: str
    time: str
    social_score: float  # From SocialReward.combined_score (raw)
    subjective_score: float  # From SubjectiveReward.score (raw)
    economy_score: float  # Deposit delta over the past year (raw)
    total_score: float  # Weighted combination of z-scores

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TotalReward":
        return TotalReward(
            agent_name=d["agent_name"],
            time=d["time"],
            social_score=d["social_score"],
            subjective_score=d["subjective_score"],
            economy_score=d["economy_score"],
            total_score=d["total_score"],
        )


# =============================================================================
# Ranking → Weight Conversion
# =============================================================================


def ranking_to_weights(
    ranking: List[str],
    max_score: float = 100.0,
) -> Dict[str, float]:
    """Convert ranking list to weights with uniform distribution.

    Args:
        ranking: List of names from highest to lowest rank
        max_score: Score for 1st place (last place gets 0)

    Returns:
        Dict mapping name to weight

    Rules:
        - 1st place: max_score (100)
        - Last place: 0
        - Middle: uniformly distributed
        - Empty list: empty dict
    """
    if not ranking:
        return {}

    n = len(ranking)
    if n == 1:
        return {ranking[0]: max_score}

    step = max_score / (n - 1)
    return {name: max_score - i * step for i, name in enumerate(ranking)}


# =============================================================================
# PageRank Algorithm
# =============================================================================


def pagerank(
    graph: SocialGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    mutual_affection_alpha: float = 0.0,
) -> Dict[str, float]:
    """Compute PageRank scores for a weighted directed graph.

    Args:
        graph: Dict[from_agent, Dict[to_agent, weight]]
        damping: Damping factor (standard: 0.85)
        max_iter: Maximum iterations
        tol: Convergence threshold
        mutual_affection_alpha: If > 0, apply mutual affection bonus after convergence.
            The formula becomes: S_I' = Σ_j (w_ji × (1 + α × w_ij) × S_j)
            This rewards agents whose liked ones also like them back.

    Returns:
        Dict[agent_name, score], scores normalized to [0, 1]
    """
    # 1. Collect all nodes
    nodes = set(graph.keys())
    for targets in graph.values():
        nodes.update(targets.keys())
    nodes = sorted(nodes)  # Deterministic ordering
    n = len(nodes)

    if n == 0:
        return {}

    # 2. Initialize scores (uniform distribution)
    scores = {node: 1.0 / n for node in nodes}

    # 3. Normalize outgoing edge weights
    out_weights: Dict[str, Dict[str, float]] = {}
    for src, targets in graph.items():
        total = sum(targets.values())
        if total > 0:
            out_weights[src] = {t: w / total for t, w in targets.items()}
        else:
            out_weights[src] = {}

    # 4. Iterate until convergence
    for _ in range(max_iter):
        new_scores = {node: (1 - damping) / n for node in nodes}

        for src in nodes:
            if src not in out_weights or not out_weights[src]:
                # Dangling node (no outgoing edges): distribute score to all nodes
                # This corresponds to PageRank's teleportation mechanism
                for node in nodes:
                    new_scores[node] += damping * scores[src] / n
            else:
                for tgt, weight in out_weights[src].items():
                    new_scores[tgt] += damping * scores[src] * weight

        # Check convergence
        diff = sum(abs(new_scores[node] - scores[node]) for node in nodes)
        scores = new_scores
        if diff < tol:
            break

    # 5. Apply mutual affection bonus (post-convergence, single pass)
    # Formula: S_I' = Σ_j (w_ji × (1 + α × w_ij) × S_j)
    # Effect: If I likes J (high w_ij), I gets bonus from J's contribution
    if mutual_affection_alpha > 0:
        new_scores = {node: 0.0 for node in nodes}

        for src in nodes:
            for tgt, weight in out_weights.get(src, {}).items():
                # weight = w_{src->tgt} (src likes tgt)
                # tgt gets bonus if tgt also likes src
                tgt_likes_src = out_weights.get(tgt, {}).get(src, 0.0)
                multiplier = 1 + mutual_affection_alpha * tgt_likes_src
                new_scores[tgt] += scores[src] * weight * multiplier

        scores = new_scores

    return scores


# =============================================================================
# Graph Construction
# =============================================================================


def build_social_graphs(
    rankings: List[SocialRanking],
) -> Tuple[SocialGraph, SocialGraph]:
    """Build affection and respect graphs from ranking data.

    Args:
        rankings: List of all agents' ranking data

    Returns:
        (affection_graph, respect_graph)
    """
    affection_graph: SocialGraph = {}
    respect_graph: SocialGraph = {}

    for ranking in rankings:
        agent = ranking.agent_name

        # Convert rankings to weights
        affection_weights = ranking_to_weights(ranking.affection_ranking)
        respect_weights = ranking_to_weights(ranking.respect_ranking)

        # Add to graphs (only if non-empty)
        if affection_weights:
            affection_graph[agent] = affection_weights
        if respect_weights:
            respect_graph[agent] = respect_weights

    return affection_graph, respect_graph


# =============================================================================
# Social Reward Calculation
# =============================================================================


def calculate_social_rewards(
    affection_graph: SocialGraph,
    respect_graph: SocialGraph,
    time_str: str,
    all_agent_names: Optional[List[str]] = None,
) -> Dict[str, SocialReward]:
    """Calculate social achievement for all agents (from others' evaluation).

    Args:
        affection_graph: Affection weighted graph
        respect_graph: Respect weighted graph
        time_str: Current time string
        all_agent_names: If provided, ensures all these agents appear in results
                         (with 0.0 scores for those not in graphs)

    Returns:
        Dict[agent_name, SocialReward]
    """
    from src.utils import get_verify_logger

    verify_logger = get_verify_logger(feature="reward")

    config = get_config()
    reward_cfg = config["world"]["reward"]
    affection_weight = reward_cfg["affection_weight"]
    damping = reward_cfg["pagerank_damping"]
    max_iter = reward_cfg["pagerank_max_iter"]
    mutual_alpha = reward_cfg["mutual_affection_alpha"]
    if verify_logger:
        verify_logger.info(
            f"[VERIFY-REWARD] calculate_social_rewards: "
            f"affection_weight={affection_weight}, damping={damping}, "
            f"max_iter={max_iter}, mutual_alpha={mutual_alpha}"
        )

    # Compute PageRank for both dimensions (with mutual affection bonus)
    affection_scores = pagerank(
        affection_graph,
        damping=damping,
        max_iter=max_iter,
        mutual_affection_alpha=mutual_alpha,
    )
    respect_scores = pagerank(
        respect_graph,
        damping=damping,
        max_iter=max_iter,
        mutual_affection_alpha=mutual_alpha,
    )

    if verify_logger:
        # Log PageRank top scores
        aff_top = sorted(affection_scores.items(), key=lambda x: -x[1])[:5]
        resp_top = sorted(respect_scores.items(), key=lambda x: -x[1])[:5]
        verify_logger.info(
            f"[VERIFY-REWARD] PageRank affection top5: "
            f"{[(n, f'{s:.3f}') for n, s in aff_top]}"
        )
        verify_logger.info(
            f"[VERIFY-REWARD] PageRank respect top5: "
            f"{[(n, f'{s:.3f}') for n, s in resp_top]}"
        )

    # Determine agent set: if explicit list provided, use it as the authoritative set
    # (PageRank graphs may contain spurious names hallucinated by LLM)
    if all_agent_names is not None:
        all_agents = set(all_agent_names)
    else:
        all_agents = set(affection_scores.keys()) | set(respect_scores.keys())

    # Build SocialReward for each agent
    results: Dict[str, SocialReward] = {}
    respect_weight = 1.0 - affection_weight

    for agent in sorted(all_agents):
        # Agent may only appear in one graph (or neither if from all_agent_names)
        # This is expected: new agents or isolated agents have 0 score
        aff_score = affection_scores[agent] if agent in affection_scores else 0.0
        resp_score = respect_scores[agent] if agent in respect_scores else 0.0
        combined = affection_weight * aff_score + respect_weight * resp_score

        results[agent] = SocialReward(
            agent_name=agent,
            time=time_str,
            affection_score=aff_score,
            respect_score=resp_score,
            combined_score=combined,
        )

    return results


# =============================================================================
# LLM-based Ranking Generation
# =============================================================================


def _validate_ranking_response(response: str, **kwargs) -> Optional[Dict]:
    """Post-processor for social ranking LLM output.

    Expected format:
    {
        "affection": {"person_name": score, ...},
        "respect": {"person_name": score, ...}
    }

    Args:
        response: LLM output string
        **kwargs: MUST include 'known_names' (set) for validation.
                  Will raise KeyError if missing.

    Returns:
        Dict with affection_scores and respect_scores, or None to trigger retry
    """
    data = extract_json(response, **kwargs)
    if not data or not isinstance(data, dict):
        return None

    known_names: set = kwargs["known_names"]

    # Validate both dimensions exist as dicts
    if "affection" not in data or not isinstance(data["affection"], dict):
        return None
    if "respect" not in data or not isinstance(data["respect"], dict):
        return None

    def _parse_scores(raw: Dict) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for name, val in raw.items():
            if name not in known_names:
                continue
            try:
                score = int(round(float(val)))
            except (ValueError, TypeError):
                continue
            result[name] = max(0, min(100, score))
        return result

    return {
        "affection_scores": _parse_scores(data["affection"]),
        "respect_scores": _parse_scores(data["respect"]),
    }


# =============================================================================
# Cross-Run Social Metrics (absolute, non-zero-sum)
# =============================================================================


@dataclass
class SocialMetrics:
    """Absolute social metrics for a single agent (non-zero-sum, cross-run comparable)."""

    agent_name: str
    time: str
    avg_affection_from_others: float
    avg_respect_from_others: float
    num_people_favor: int  # affection >= SOCIAL_NEUTRAL_SCORE
    num_people_respect: int  # respect >= SOCIAL_NEUTRAL_SCORE
    top_1_avg_favor: float
    top_3_avg_favor: float
    top_10_avg_favor: float
    top_1_avg_respect: float
    top_3_avg_respect: float
    top_10_avg_respect: float
    top_1_avg_favor_mutual: float
    top_3_avg_favor_mutual: float
    top_10_avg_favor_mutual: float
    top_1_avg_respect_mutual: float
    top_3_avg_respect_mutual: float
    top_10_avg_respect_mutual: float


def compute_social_metrics(
    rankings: List[SocialRanking],
    time_str: str,
) -> Dict[str, SocialMetrics]:
    """Compute absolute social metrics from rankings (pure function).

    Builds received/given score indices, then computes per-agent metrics.
    """
    all_names = sorted({r.agent_name for r in rankings})

    # Build indices: received[target][source] = score
    aff_received: Dict[str, Dict[str, int]] = {n: {} for n in all_names}
    resp_received: Dict[str, Dict[str, int]] = {n: {} for n in all_names}
    # Given: given[source][target] = score (same as ranking scores)
    aff_given: Dict[str, Dict[str, int]] = {}
    resp_given: Dict[str, Dict[str, int]] = {}

    for r in rankings:
        aff_given[r.agent_name] = r.affection_scores
        resp_given[r.agent_name] = r.respect_scores
        for target, score in r.affection_scores.items():
            if target in aff_received:
                aff_received[target][r.agent_name] = score
        for target, score in r.respect_scores.items():
            if target in resp_received:
                resp_received[target][r.agent_name] = score

    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _top_k_avg(scores: Dict[str, int], k: int) -> float:
        """Average of top-k scores received FROM others (who likes this agent most)."""
        if not scores:
            return 0.0
        sorted_targets = sorted(scores.keys(), key=lambda n: (-scores[n], n))
        top = sorted_targets[:k]
        return _avg([scores[t] for t in top])

    def _top_k_mutual(
        given_scores: Dict[str, int],
        received_from: Dict[str, int],
        k: int,
    ) -> float:
        """For agent's top-k favorites, average of how much those people like agent back."""
        if not given_scores:
            return 0.0
        sorted_targets = sorted(
            given_scores.keys(), key=lambda n: (-given_scores[n], n)
        )
        top = sorted_targets[:k]
        return _avg([received_from.get(t, SOCIAL_NEUTRAL_SCORE) for t in top])

    results: Dict[str, SocialMetrics] = {}
    for name in all_names:
        aff_from = aff_received[name]
        resp_from = resp_received[name]
        aff_to = aff_given[name]
        resp_to = resp_given[name]

        results[name] = SocialMetrics(
            agent_name=name,
            time=time_str,
            avg_affection_from_others=_avg(list(aff_from.values())),
            avg_respect_from_others=_avg(list(resp_from.values())),
            num_people_favor=sum(
                1 for v in aff_from.values() if v >= SOCIAL_NEUTRAL_SCORE
            ),
            num_people_respect=sum(
                1 for v in resp_from.values() if v >= SOCIAL_NEUTRAL_SCORE
            ),
            top_1_avg_favor=_top_k_avg(aff_from, 1),
            top_3_avg_favor=_top_k_avg(aff_from, 3),
            top_10_avg_favor=_top_k_avg(aff_from, 10),
            top_1_avg_respect=_top_k_avg(resp_from, 1),
            top_3_avg_respect=_top_k_avg(resp_from, 3),
            top_10_avg_respect=_top_k_avg(resp_from, 10),
            top_1_avg_favor_mutual=_top_k_mutual(aff_to, aff_received[name], 1),
            top_3_avg_favor_mutual=_top_k_mutual(aff_to, aff_received[name], 3),
            top_10_avg_favor_mutual=_top_k_mutual(aff_to, aff_received[name], 10),
            top_1_avg_respect_mutual=_top_k_mutual(resp_to, resp_received[name], 1),
            top_3_avg_respect_mutual=_top_k_mutual(resp_to, resp_received[name], 3),
            top_10_avg_respect_mutual=_top_k_mutual(resp_to, resp_received[name], 10),
        )

    return results


def save_social_metrics(
    metrics: Dict[str, SocialMetrics],
    data_dir: str,
    year: int,
    week: int | None = None,
    day: int | None = None,
) -> None:
    """Save social metrics to reward/metrics/year=Y/[week=W|day=D].jsonl.

    Supports both weekly and daily storage.

    Args:
        metrics: Dict mapping agent name to SocialMetrics
        data_dir: World data directory name
        year: Year number
        week: Week number (for weekly storage)
        day: Day number (for daily storage)
    """
    _save_reward_jsonl(list(metrics.values()), data_dir, "metrics", year, week, day)


# =============================================================================
# Persistence Helpers (for global aggregate data like advantages)
# =============================================================================


def _save_reward_jsonl(
    items: List,
    data_dir: str,
    subdir: str,
    year: int,
    week: int | None = None,
    day: int | None = None,
) -> None:
    """Write dataclass items to JSONL file.

    Supports both weekly and daily storage:
    - Weekly: reward/{subdir}/year=Y/week=W.jsonl
    - Daily:  reward/{subdir}/year=Y/day=D.jsonl

    Args:
        items: List of dataclass objects
        data_dir: World data directory name
        subdir: Subdirectory under reward/ (e.g., "rankings", "metrics")
        year: Year number
        week: Week number (for weekly storage)
        day: Day number (for daily storage)
    """
    base = Path("data") / data_dir / "reward" / subdir / f"year={year}"
    if day is not None:
        path = base / f"day={day}.jsonl"
    else:
        path = base / f"week={week}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def save_rankings(
    rankings: List[SocialRanking],
    data_dir: str,
    year: int,
    week: int | None = None,
    day: int | None = None,
) -> None:
    """Save all agents' rankings to centralized location.

    Supports both weekly and daily storage:
    - Weekly: data/{world}/reward/rankings/year=YYYY/week=W.jsonl
    - Daily:  data/{world}/reward/rankings/year=YYYY/day=D.jsonl

    This is the PageRank input data - needed to reconstruct social rewards.
    Per-agent reward.jsonl only stores computed results, not inputs.

    Args:
        rankings: List of SocialRanking objects
        data_dir: World data directory name
        year: Year number
        week: Week number (for weekly storage)
        day: Day number (for daily storage)
    """
    _save_reward_jsonl(rankings, data_dir, "rankings", year, week, day)


def load_rankings(
    data_dir: str,
    year: int,
    week: int | None = None,
    day: int | None = None,
) -> List[SocialRanking]:
    """Load rankings from centralized storage.

    Supports both weekly and daily storage.

    Args:
        data_dir: World data directory name
        year: Year number
        week: Week number (for weekly storage)
        day: Day number (for daily storage)

    Returns:
        List of SocialRanking for all agents at the specified time.
    """
    if day is not None:
        path = (
            Path("data")
            / data_dir
            / "reward"
            / "rankings"
            / f"year={year}"
            / f"day={day}.jsonl"
        )
    else:
        path = (
            Path("data")
            / data_dir
            / "reward"
            / "rankings"
            / f"year={year}"
            / f"week={week}.jsonl"
        )

    if not path.exists():
        return []

    rankings = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rankings.append(SocialRanking.from_dict(d))
    return rankings


# =============================================================================
# Daily to Weekly Normalization
# =============================================================================

def normalize_daily_to_weekly(
    data_dir: str,
    year: int,
    week: int,
    days: List[int],
) -> None:
    """Aggregate N days of daily reward data into weekly reward data.

    Normalization methods by reward type:
    - Social reward (combined_score): mean across days
    - Subjective reward (score): mean across days
    - Economy reward (score): sum across days (cumulative)

    Args:
        data_dir: World data directory name
        year: Year number
        week: Week number to generate
        days: List of day numbers to aggregate (e.g., [1, 2, 3, 4, 5])
    """
    from src.utils import get_logger
    logger = get_logger("reward")

    # This function aggregates daily reward data into weekly for display/analysis.
    # Daily data is saved to reward/{subdir}/year=Y/day=D.jsonl
    # Weekly data is saved to reward/{subdir}/year=Y/week=W.jsonl
    #
    # Implementation notes:
    # 1. Read daily rankings from day=1..N
    # 2. Average social scores across days
    # 3. Read daily per-agent rewards
    # 4. Aggregate: mean for social/subjective, sum for economy
    # 5. Save as weekly data

    logger.info(f"Normalizing daily data to weekly: year={year} week={week} days={days}")
    logger.warning("normalize_daily_to_weekly: Not yet fully implemented - daily data saved, manual aggregation needed for weekly view")


# =============================================================================
# Subjective Reward Calculation
# =============================================================================

# Fulfillment dimensions used across the system
FULFILLMENT_DIMS = ["mood", "material", "social", "esteem"]


def compute_subjective_rewards(
    agents: List["RoleAgent"],
    time_str: str,
) -> Dict[str, SubjectiveReward]:
    """Compute subjective rewards for all agents (pure data-driven).

    1. Collect all fulfillment values across agents × weeks, separated by dimension
    2. Compute per-dimension threshold (bottom percentile of each dimension's values)
    3. For each agent: apply penalty to values below dimension threshold, then compute mean

    Args:
        agents: List of all RoleAgent instances
        time_str: Current time string

    Returns:
        Dict[agent_name, SubjectiveReward]
    """
    from src.utils import get_verify_logger

    verify_logger = get_verify_logger(feature="reward")

    config = get_config()
    reward_cfg = config["world"]["reward"]
    granularity = reward_cfg.get("granularity", "weekly")
    if granularity == "daily":
        n_periods = reward_cfg.get("period_days", 5)
    else:
        n_periods = reward_cfg.get("period_weeks", 1)
    percentile = reward_cfg["misery_threshold_percentile"]
    penalty_value = reward_cfg["misery_penalty_value"]

    if verify_logger:
        verify_logger.info(
            f"[VERIFY-REWARD] compute_subjective_rewards: "
            f"granularity={granularity}, n_periods={n_periods}, "
            f"percentile={percentile}, penalty={penalty_value}"
        )

    # Phase 1: Collect all data, separated by dimension (fulfillment + vitality)
    agent_histories: Dict[str, List[Dict]] = {}
    all_values_by_dim: Dict[str, List[float]] = {dim: [] for dim in FULFILLMENT_DIMS}
    all_vitality_values: List[float] = []

    for agent in agents:
        if granularity == "daily":
            history = agent.dm.get_fulfillment_history_daily(n_periods=n_periods)
        else:
            history = agent.dm.get_fulfillment_history(n_weeks=n_periods)
        agent_histories[agent.name] = history
        for entry in history:
            for dim in FULFILLMENT_DIMS:
                all_values_by_dim[dim].append(entry["fulfillment"][dim])
            all_vitality_values.append(entry["vitality"])

    if verify_logger:
        for name in sorted(agent_histories.keys()):
            verify_logger.info(
                f"[VERIFY-REWARD] {name} fulfillment_history: "
                f"{len(agent_histories[name])} entries"
            )

    # Phase 2: Compute per-dimension threshold (fulfillment + vitality)
    thresholds: Dict[str, float] = {}
    for dim in FULFILLMENT_DIMS:
        values = all_values_by_dim[dim]
        if len(set(values)) <= 1:
            # All same value or empty - no penalty possible for this dimension
            thresholds[dim] = float("-inf")
        else:
            sorted_values = sorted(values)
            idx = max(0, int(len(sorted_values) * percentile) - 1)
            thresholds[dim] = sorted_values[idx]

    # Vitality threshold (separate from fulfillment dimensions)
    if len(set(all_vitality_values)) <= 1:
        vitality_threshold = float("-inf")
    else:
        sorted_vitality = sorted(all_vitality_values)
        idx = max(0, int(len(sorted_vitality) * percentile) - 1)
        vitality_threshold = sorted_vitality[idx]

    if verify_logger:
        threshold_str = ", ".join(
            f"{dim}={thresholds[dim]:.2f}" for dim in FULFILLMENT_DIMS
        )
        verify_logger.info(
            f"[VERIFY-REWARD] misery thresholds: {threshold_str}, "
            f"vitality={vitality_threshold:.2f}"
        )

    # Phase 3: Compute SubjectiveReward for each agent
    results: Dict[str, SubjectiveReward] = {}
    for agent in agents:
        name = agent.name
        history = agent_histories[name]
        if not history:
            raise ValueError(f"Agent '{name}' has no fulfillment history - data error")

        # Apply penalty to fulfillment values below dimension threshold, compute mean
        adjusted_values: List[float] = []
        n_penalties = 0

        for entry in history:
            for dim in FULFILLMENT_DIMS:
                val = entry["fulfillment"][dim]
                if val < thresholds[dim]:
                    adjusted_values.append(val - penalty_value)
                    n_penalties += 1
                else:
                    adjusted_values.append(val)

        # Vitality penalty: doesn't contribute to mean, only applies penalty
        # Each low vitality instance subtracts penalty_value / base_count from score
        n_vitality_penalties = 0
        for entry in history:
            if entry["vitality"] < vitality_threshold:
                n_vitality_penalties += 1

        base_count = len(adjusted_values)  # fulfillment data points
        score = sum(adjusted_values) / base_count
        # Apply vitality penalty (same magnitude as fulfillment penalty)
        score -= n_vitality_penalties * penalty_value / base_count
        n_penalties += n_vitality_penalties

        results[name] = SubjectiveReward(
            agent_name=name,
            time=time_str,
            score=score,
            n_penalties=n_penalties,
        )

        if verify_logger and n_penalties > 0:
            verify_logger.info(
                f"[VERIFY-REWARD] {name} subjective: "
                f"score={score:.3f}, penalties={n_penalties} "
                f"(vitality={n_vitality_penalties})"
            )

    return results


# =============================================================================
# Total Reward Calculation
# =============================================================================


def calculate_total_rewards(
    social_rewards: Dict[str, SocialReward],
    subjective_rewards: Dict[str, SubjectiveReward],
    economy_scores: Dict[str, float],
    time_str: str,
) -> Dict[str, TotalReward]:
    """Combine social, subjective, and economy rewards into total reward.

    All three components are z-score normalized before combining.
    This ensures all components have comparable scale (mean=0, std=1) regardless
    of their original distributions.

    Precondition: social_rewards, subjective_rewards, and economy_scores must
    contain the same set of agent names. If not, KeyError will be raised.

    Args:
        social_rewards: Dict of SocialReward per agent
        subjective_rewards: Dict of SubjectiveReward per agent
        economy_scores: Dict mapping agent_name to deposit delta (economy change)
        time_str: Current time string

    Returns:
        Dict[agent_name, TotalReward]
    """
    config = get_config()
    reward_cfg = config["world"]["reward"]
    social_weight = reward_cfg["social_weight"]
    economy_weight = reward_cfg["economy_weight"]
    # Remaining weight goes to subjective
    subjective_weight = 1.0 - social_weight - economy_weight

    # Validate agent sets match
    social_agents = set(social_rewards.keys())
    subj_agents = set(subjective_rewards.keys())
    econ_agents = set(economy_scores.keys())
    if social_agents != subj_agents or social_agents != econ_agents:
        raise ValueError(
            f"Agent set mismatch: social={len(social_agents)}, "
            f"subjective={len(subj_agents)}, economy={len(econ_agents)}"
        )

    # Collect raw scores
    social_scores = [r.combined_score for r in social_rewards.values()]
    subj_scores = [r.score for r in subjective_rewards.values()]
    econ_scores = list(economy_scores.values())

    # Helper to compute z-score stats
    def z_stats(scores: List[float]) -> Tuple[float, float]:
        if not scores:
            return 0.0, 1.0
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = var**0.5 if var > 0 else 1.0
        return mean, std

    social_mean, social_std = z_stats(social_scores)
    subj_mean, subj_std = z_stats(subj_scores)
    econ_mean, econ_std = z_stats(econ_scores)

    results: Dict[str, TotalReward] = {}

    for agent in sorted(social_rewards.keys()):
        raw_social = social_rewards[agent].combined_score
        raw_subj = subjective_rewards[agent].score
        raw_econ = economy_scores[agent]

        # Z-score normalize for combining (scale alignment)
        social_z = (raw_social - social_mean) / social_std
        subj_z = (raw_subj - subj_mean) / subj_std
        econ_z = (raw_econ - econ_mean) / econ_std

        # Weighted combination of z-scores
        total = (
            social_weight * social_z
            + subjective_weight * subj_z
            + economy_weight * econ_z
        )

        # Store raw scores (for analysis), but total uses z-score normalized values
        results[agent] = TotalReward(
            agent_name=agent,
            time=time_str,
            social_score=raw_social,
            subjective_score=raw_subj,
            economy_score=raw_econ,
            total_score=total,
        )

    return results


# =============================================================================
# Return and Advantage Calculation (for RL Optimization)
# =============================================================================


def calculate_returns(
    reward_history: Dict[str, List[Tuple[str, float]]],
    normalize: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """Calculate discounted returns for each agent at each time point.

    Return_t = sum_{k=0}^{T-t} gamma^k * reward_{t+k}

    In practice, we compute from the end backwards:
    Return_T = reward_T
    Return_{t} = reward_t + gamma * Return_{t+1}

    IMPORTANT: We prepend a virtual time point (year_begin, reward=0) before the
    first reward. This ensures that the first period's behavior has an advantage:

    Timeline:  |----Year 1----|----Year 2----|----Year 3----|
               ^              ^              ^              ^
            Return_0       Return_1       Return_2       Return_3
            (W00-begin)    (settle)       (settle)       (settle)
            reward=0       reward=r₁      reward=r₂      reward=r₃

    This allows advantage[0] = Return_1 - Return_0 to measure Year 1's behavior.

    When normalize=True, we divide by the discounted "effective steps" to make
    returns at different time points comparable in scale:
    Return_t_normalized = Return_t / (1 + γ + γ² + … + γ^{T-t})
                        = Return_t / [(1 - γ^{T-t+1}) / (1 - γ)]

    This ensures that if all rewards are constant r, then Return_t_normalized = r
    for all t, eliminating the scale difference between early and late time points.

    Args:
        reward_history: Dict[agent, List[(time, reward)]] sorted by time
        normalize: If True, normalize returns by discounted effective steps

    Returns:
        Dict[agent, List[(time, return)]] with computed (optionally normalized) returns
        Note: The first element is the virtual year-begin point (reward=0)
    """
    config = get_config()
    gamma = config["world"]["reward"]["gamma"]

    results: Dict[str, List[Tuple[str, float]]] = {}

    for agent, history in reward_history.items():
        if not history:
            raise ValueError(f"Agent '{agent}' has empty reward history")

        # Extract the start year from first reward time
        first_time_str = history[0][0]
        start_year = TimeState.from_string(first_time_str).year

        # Prepend virtual (year_begin, reward=0) record
        year_begin_time = TimeState.get_year_begin(start_year)
        extended_history = [(year_begin_time, 0.0)] + list(history)

        n = len(extended_history)
        returns = [0.0] * n

        # Backward pass to compute raw returns
        returns[n - 1] = extended_history[n - 1][1]
        for i in range(n - 2, -1, -1):
            returns[i] = extended_history[i][1] + gamma * returns[i + 1]

        # Normalize by discounted effective steps if requested
        # Note: Skip when gamma=1.0 to avoid division by zero in (1-γ)
        if normalize and gamma < 1.0:
            for i in range(n):
                remaining_steps = n - i  # T - t + 1
                # Discounted effective steps: (1 - γ^remaining_steps) / (1 - γ)
                effective_steps = (1 - gamma**remaining_steps) / (1 - gamma)
                returns[i] = returns[i] / effective_steps

        results[agent] = [(extended_history[i][0], returns[i]) for i in range(n)]

    return results


def calculate_advantages(
    returns: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, List[Tuple[str, str, float]]]:
    """Calculate advantages from returns over a time period.

    Advantage measures state improvement: A_{period} = Return_end - Return_start

    Timeline and advantage mapping:
        Timeline:  |----Year 1----|----Year 2----|----Year 3----|
                   ^              ^              ^              ^
                Return_0       Return_1       Return_2       Return_3
               (W00-begin)    (settle)       (settle)       (settle)

        advantage[0]: start=Y2020-W00-begin, end=Y2020-W01-settle (Year 1 behavior)
        advantage[1]: start=Y2021-W00-begin, end=Y2021-W01-settle (Year 2 behavior)
        advantage[2]: start=Y2022-W00-begin, end=Y2022-W01-settle (Year 3 behavior)

    The time period format:
    - start: beginning of the year (e.g., "Y2020-W00-begin")
    - end: settle point of the year's last week (e.g., "Y2020-W10-settle")

    Note: Per-period z-score normalization was removed because reward-level z-score
    (applied before merging social/subjective) already ensures E[reward]=0 at each
    timestep, which makes E[advantage]=0 as well. The slight variance bias in the first
    timestep (A_0 = Return_1 - 0) is acceptable.

    Args:
        returns: Dict[agent, List[(time, return)]] sorted by time
                 First element is year-begin (W00-begin), rest are settle points

    Returns:
        Dict[agent, List[(start_time, end_time, advantage)]]
    """
    results: Dict[str, List[Tuple[str, str, float]]] = {}

    for agent, ret_list in returns.items():
        if len(ret_list) < 2:
            continue

        advantages = []
        for i in range(1, len(ret_list)):
            # start_time: W00-begin for first, settle -> W00-begin of next year for rest
            start_time = ret_list[i - 1][0]
            if "-settle" in start_time:
                # "Y2020-W10-settle" -> "Y2021-W00-begin"
                year = TimeState.from_string(start_time).year
                start_time = TimeState.get_year_begin(year + 1)

            # end_time: keep settle format as-is (e.g., "Y2020-W10-settle")
            end_time = ret_list[i][0]

            advantage = ret_list[i][1] - ret_list[i - 1][1]
            advantages.append((start_time, end_time, advantage))

        results[agent] = advantages

    return results


def select_top_trajectories(
    advantages: Dict[str, List[Tuple[str, str, float]]],
    top_fraction: float = 1.0 / 3.0,
) -> List[Tuple[str, str, str, float]]:
    """Select top trajectories by advantage for training.

    Args:
        advantages: Dict[agent, List[(start_time, end_time, advantage)]]
        top_fraction: Fraction of trajectories to select (default 1/3)

    Returns:
        List[(agent_name, start_time, end_time, advantage)] for top trajectories
    """
    # Flatten all (agent, start, end, advantage) tuples
    all_advantages: List[Tuple[str, str, str, float]] = []
    for agent, adv_list in advantages.items():
        for start_time, end_time, advantage in adv_list:
            all_advantages.append((agent, start_time, end_time, advantage))

    if not all_advantages:
        return []

    # Sort by advantage (descending), with tie-breaker for determinism
    all_advantages.sort(key=lambda x: (-x[3], x[0], x[1], x[2]))

    # Select top fraction
    n_select = max(1, int(len(all_advantages) * top_fraction))
    return all_advantages[:n_select]


def save_advantages(
    advantages: Dict[str, List[Tuple[str, str, float]]],
    data_dir: str,
) -> None:
    """Save advantage scores to JSONL file.

    Path: data/{world}/reward/advantages.jsonl

    Format:
    {
        "agent_name": "Bei",
        "time": {"start": "Y2020-W00-begin", "end": "Y2020-W10-settle"},
        "advantage": 9.2
    }
    """
    path = Path("data") / data_dir / "reward" / "advantages.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for agent in sorted(advantages.keys()):
            for start_time, end_time, advantage in advantages[agent]:
                entry = {
                    "agent_name": agent,
                    "time": {"start": start_time, "end": end_time},
                    "advantage": advantage,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
