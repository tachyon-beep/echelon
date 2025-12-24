from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_GLICKO2_SCALE = 173.7178
_DEFAULT_RATING = 1500.0


@dataclass(frozen=True)
class Glicko2Config:
    tau: float = 0.5
    epsilon: float = 1e-6
    rating0: float = _DEFAULT_RATING
    rd0: float = 350.0
    vol0: float = 0.06


@dataclass(frozen=True)
class Glicko2Rating:
    rating: float = _DEFAULT_RATING
    rd: float = 350.0
    vol: float = 0.06

    @property
    def conservative_rating(self) -> float:
        """Rating minus 2 standard deviations (95% lower bound).

        Use this for matchmaking to avoid overrating uncertain players.
        """
        return float(self.rating) - 2.0 * float(self.rd)

    def as_dict(self) -> dict[str, float]:
        return {
            "rating": float(self.rating),
            "rd": float(self.rd),
            "vol": float(self.vol),
            "conservative_rating": self.conservative_rating,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Glicko2Rating:
        return cls(rating=float(d["rating"]), rd=float(d["rd"]), vol=float(d["vol"]))

    def with_defaults(self, cfg: Glicko2Config) -> Glicko2Rating:
        # Useful if older league files are missing some fields.
        return dataclasses.replace(
            self,
            rating=float(self.rating if self.rating is not None else cfg.rating0),
            rd=float(self.rd if self.rd is not None else cfg.rd0),
            vol=float(self.vol if self.vol is not None else cfg.vol0),
        )


@dataclass(frozen=True)
class GameResult:
    opponent: Glicko2Rating
    score: float  # 1.0 win, 0.5 draw, 0.0 loss


def _to_mu_phi(r: Glicko2Rating, cfg: Glicko2Config) -> tuple[float, float, float]:
    mu = (float(r.rating) - float(cfg.rating0)) / _GLICKO2_SCALE
    phi = float(r.rd) / _GLICKO2_SCALE
    sigma = float(r.vol)
    return mu, phi, sigma


def _to_rating(mu: float, phi: float, sigma: float, cfg: Glicko2Config) -> Glicko2Rating:
    rating = _GLICKO2_SCALE * mu + float(cfg.rating0)
    rd = _GLICKO2_SCALE * phi
    return Glicko2Rating(rating=float(rating), rd=float(rd), vol=float(sigma))


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * (phi**2) / (math.pi**2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _solve_sigma(
    *,
    phi: float,
    sigma: float,
    delta: float,
    v: float,
    tau: float,
    epsilon: float,
) -> float:
    a = math.log(sigma**2)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta**2 - phi**2 - v - ex)
        den = 2.0 * (phi**2 + v + ex) ** 2
        return (num / den) - ((x - a) / (tau**2))

    A = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
        B = a - k * tau

    fA = f(A)
    fB = f(B)

    # Regula falsi with Illinois modification (as in the original Glicko-2 paper).
    while abs(B - A) > epsilon:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0.0:
            A = B
            fA = fB
        else:
            fA *= 0.5
        B = C
        fB = fC

    return float(math.exp(A / 2.0))


def rate(
    rating: Glicko2Rating,
    results: Iterable[GameResult],
    *,
    cfg: Glicko2Config = Glicko2Config(),
) -> Glicko2Rating:
    results = list(results)
    mu, phi, sigma = _to_mu_phi(rating, cfg)

    if not results:
        # No games: keep mu/sigma, increase phi due to uncertainty.
        phi_prime = math.sqrt(phi**2 + sigma**2)
        return _to_rating(mu, phi_prime, sigma, cfg)

    opp = [(_to_mu_phi(r.opponent, cfg), float(r.score)) for r in results]
    g_list = [_g(phi_j) for (mu_j, phi_j, _), _ in opp]
    E_list = [_E(mu, mu_j, phi_j) for (mu_j, phi_j, _), _ in opp]

    v_inv = 0.0
    for g_j, E_j in zip(g_list, E_list, strict=True):
        v_inv += (g_j**2) * E_j * (1.0 - E_j)
    v = 1.0 / max(1e-12, v_inv)

    delta_sum = 0.0
    for ((_mu_j, _phi_j, _), s_j), g_j, E_j in zip(opp, g_list, E_list, strict=True):
        delta_sum += g_j * (s_j - E_j)
    delta = v * delta_sum

    sigma_prime = _solve_sigma(
        phi=phi,
        sigma=sigma,
        delta=delta,
        v=v,
        tau=float(cfg.tau),
        epsilon=float(cfg.epsilon),
    )
    phi_star = math.sqrt(phi**2 + sigma_prime**2)
    phi_prime = 1.0 / math.sqrt((1.0 / (phi_star**2)) + (1.0 / v))

    mu_prime = mu + (phi_prime**2) * delta_sum
    return _to_rating(mu_prime, phi_prime, sigma_prime, cfg)


def expected_score(a: Glicko2Rating, b: Glicko2Rating, *, cfg: Glicko2Config = Glicko2Config()) -> float:
    mu_a, _, _ = _to_mu_phi(a, cfg)
    mu_b, phi_b, _ = _to_mu_phi(b, cfg)
    return float(_E(mu_a, mu_b, phi_b))
