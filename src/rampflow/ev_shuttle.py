from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import Iterable

from .ledger import ConflictError, Reservation, ReservationLedger, StaleGeneration
from .models import Decision
from .timeutil import Window, subtract_windows, utc


@dataclass(frozen=True)
class Charger:
    """A depot charging point.  Fast chargers are the scarce resource that plans
    try to leave idle unless a slow charger cannot finish before departure."""

    charger_id: str
    fast: bool
    pct_per_min: float
    active: bool = True

    def __post_init__(self):
        if self.pct_per_min <= 0:
            raise ValueError("charge rate must be positive")


@dataclass(frozen=True)
class ElectricBus:
    bus_id: str
    seats: int
    standing: int
    accessible_spaces: int
    soc_pct: float
    return_reserve_pct: float = 20.0
    travel_cost_per_min: float = 0.55
    dwell_cost_per_min: float = 0.08

    def __post_init__(self):
        if not 0.0 < self.soc_pct <= 100.0:
            raise ValueError("soc must be within (0, 100]")
        if not 0.0 <= self.return_reserve_pct < 100.0:
            raise ValueError("return reserve must be within [0, 100)")
        if self.travel_cost_per_min < 0 or self.dwell_cost_per_min < 0:
            raise ValueError("energy costs cannot be negative")

    def capacity(self, accessible_needed: int = 0) -> int:
        if accessible_needed > self.accessible_spaces:
            return 0
        return self.seats + self.standing - accessible_needed


@dataclass(frozen=True)
class ShuttleMissionRequest:
    request_id: str
    flight_id: str
    passengers: int
    accessible_passengers: int
    stand: str
    terminal: str
    ready_at: datetime
    latest_arrival: datetime
    travel_min: int
    dwell_min: int = 3

    def __post_init__(self):
        object.__setattr__(self, "ready_at", utc(self.ready_at))
        object.__setattr__(self, "latest_arrival", utc(self.latest_arrival))
        if self.passengers <= 0 or self.accessible_passengers < 0:
            raise ValueError("invalid passenger counts")
        if self.accessible_passengers > self.passengers:
            raise ValueError("accessible passengers exceed total")
        if self.travel_min <= 0 or self.dwell_min < 0:
            raise ValueError("invalid trip timing")
        if self.latest_arrival <= self.ready_at:
            raise ValueError("latest arrival must follow ready time")


@dataclass(frozen=True)
class PlannedCharge:
    charger_id: str
    bus_id: str
    window: Window
    soc_before: float
    soc_after: float
    fast: bool


@dataclass(frozen=True)
class EVShuttleTrip:
    bus_id: str
    trip_no: int
    passengers: int
    accessible: int
    outbound: Window
    return_window: Window
    soc_depart: float
    soc_after: float


@dataclass(frozen=True)
class EVShuttlePlan:
    """A proposed, not-yet-locked dispatch.  Feasible plans carry the ledger
    generation they were computed against; commit validates against it."""

    request_id: str
    feasible: bool
    trips: tuple[EVShuttleTrip, ...]
    charges: tuple[PlannedCharge, ...]
    final_arrival: datetime | None
    reason: str = ""
    generation: int = -1

    @property
    def energy_shortfalls(self) -> tuple[str, ...]:
        return tuple(c.bus_id for c in self.charges)


@dataclass(frozen=True)
class EVShuttleUse:
    """Ledger payload for both bus and charger reservations."""

    kind: str  # "trip" | "charge"
    request_id: str
    bus_id: str
    soc_before: float
    soc_after: float
    charger_id: str = ""


class EVShuttlePlanner:
    """Plan-then-commit electric shuttle dispatch.

    Every feasible plan accounts for round-trip energy, the bus return reserve,
    and pre-departure charger occupancy.  ``plan`` performs no mutation;
    ``commit`` locks every bus window and charger window atomically in one
    reservation ledger, so two dispatchers racing for the same bus or charger
    cannot both win.
    """

    def __init__(
        self,
        buses: Iterable[ElectricBus],
        chargers: Iterable[Charger] = (),
        charge_horizon_min: int = 240,
        ledger: ReservationLedger[EVShuttleUse] | None = None,
    ):
        self.buses = {b.bus_id: b for b in buses}
        if len(self.buses) != len(list(buses)):
            raise ValueError("duplicate bus")
        self.chargers = {c.charger_id: c for c in chargers}
        if len(self.chargers) != len(list(chargers)):
            raise ValueError("duplicate charger")
        if charge_horizon_min <= 0:
            raise ValueError("charge horizon must be positive")
        self.charge_horizon_min = charge_horizon_min
        self.ledger = ledger or ReservationLedger[EVShuttleUse]()
        self.plans: dict[str, EVShuttlePlan] = {}

    @property
    def generation(self) -> int:
        return self.ledger.generation

    def _bus_key(self, bus_id: str) -> str:
        return f"bus:{bus_id}"

    def _charger_key(self, charger_id: str) -> str:
        return f"charger:{charger_id}"

    def _committed(self, bus_id: str) -> tuple[list[Window], list[tuple[datetime, float]]]:
        rows = self.ledger.reservations(self._bus_key(bus_id))
        windows = [r.window for r in rows]
        events = [(r.window.end, r.payload.soc_after) for r in rows]
        return windows, events

    def _earliest_free(self, blocked: list[Window], at: datetime) -> datetime:
        cur = at
        changed = True
        while changed:
            changed = False
            for w in sorted(blocked, key=lambda x: (x.start, x.end)):
                if w.start <= cur < w.end:
                    cur = w.end
                    changed = True
        return cur

    def _soc_at(self, bus: ElectricBus, events: list[tuple[datetime, float]], at: datetime) -> float:
        soc = bus.soc_pct
        for end, after in sorted(events):
            if end <= at:
                soc = after
        return soc

    def _trip_energy(self, bus: ElectricBus, travel_min: int, dwell_min: int) -> float:
        return travel_min * 2 * bus.travel_cost_per_min + dwell_min * 2 * bus.dwell_cost_per_min

    def _find_charge(
        self,
        bus: ElectricBus,
        soc_now: float,
        required: float,
        depart: datetime,
        bus_blocked: list[Window],
        charger_blocked: dict[str, list[Window]],
    ) -> tuple[Charger, Window, float] | None:
        """Earliest pre-departure charging slot that leaves the bus at or above
        ``required``.  Slow chargers win over fast ones whenever they finish in
        time, so the fast charger stays free for urgent needs."""
        deficit = required - soc_now
        if deficit <= 0:
            return None
        horizon_start = depart - timedelta(minutes=self.charge_horizon_min)
        try:
            horizon = Window(horizon_start, depart)
        except ValueError:
            return None
        candidates: list[tuple[int, datetime, int, str, Charger, Window, float]] = []
        for charger in self.chargers.values():
            if not charger.active:
                continue
            minutes = ceil(deficit / charger.pct_per_min)
            blocked = [w for w in bus_blocked if w.overlaps(horizon, touching=True)]
            blocked += [
                w for w in charger_blocked.get(charger.charger_id, []) if w.overlaps(horizon, touching=True)
            ]
            for piece in subtract_windows(horizon, blocked):
                end = piece.start + timedelta(minutes=minutes)
                if end <= piece.end:
                    soc_after = min(100.0, soc_now + minutes * charger.pct_per_min)
                    # Prefer non-fast, then earliest finish / shortest session.
                    candidates.append(
                        (1 if charger.fast else 0, end, minutes, charger.charger_id, charger, Window(piece.start, end), soc_after)
                    )
                    break
        if not candidates:
            return None
        _, _, _, _, charger, window, soc_after = min(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
        return charger, window, soc_after

    def plan(self, req: ShuttleMissionRequest) -> EVShuttlePlan:
        remaining = req.passengers
        accessible = req.accessible_passengers

        blocked_bus: dict[str, list[Window]] = {}
        events: dict[str, list[tuple[datetime, float]]] = {}
        blocked_charger: dict[str, list[Window]] = {
            cid: [r.window for r in self.ledger.reservations(self._charger_key(cid))]
            for cid in self.chargers
        }
        for bid in self.buses:
            windows, bus_events = self._committed(bid)
            blocked_bus[bid] = windows
            events[bid] = bus_events

        trips: list[EVShuttleTrip] = []
        charges: list[PlannedCharge] = []
        trip_no = 0
        energy_blocked = False

        while remaining > 0:
            options = []
            for bid, bus in self.buses.items():
                start = self._earliest_free(blocked_bus[bid], req.ready_at)
                acc = min(accessible, bus.accessible_spaces)
                cap = bus.capacity(acc)
                if cap <= 0:
                    continue
                load = min(remaining, cap)
                outbound = Window(start, start + timedelta(minutes=req.dwell_min + req.travel_min))
                return_window = Window(
                    outbound.end, outbound.end + timedelta(minutes=req.travel_min + req.dwell_min)
                )
                energy = self._trip_energy(bus, req.travel_min, req.dwell_min)
                required = energy + bus.return_reserve_pct
                if required > 100.0:
                    energy_blocked = True
                    continue
                soc_depart = self._soc_at(bus, events[bid], start)
                charge = None
                if soc_depart < required:
                    charge = self._find_charge(
                        bus, soc_depart, required, start, blocked_bus[bid], blocked_charger
                    )
                    if charge is None:
                        energy_blocked = True
                        continue
                    _, charge_window, charged_soc = charge
                    soc_depart = charged_soc
                soc_after = soc_depart - energy
                options.append((outbound.end, -load, bid, load, acc, outbound, return_window, charge, soc_depart, soc_after))

            if not options:
                reason = "charger-capacity" if energy_blocked else "fleet-capacity"
                final = trips[-1].outbound.end if trips else req.ready_at
                plan = EVShuttlePlan(req.request_id, False, tuple(trips), tuple(charges), final, reason)
                object.__setattr__(plan, "_bookings", ())
                return plan

            options.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, bid, load, acc, outbound, return_window, charge, soc_depart, soc_after = options[0]
            trip_no += 1
            trips.append(EVShuttleTrip(bid, trip_no, load, acc, outbound, return_window, soc_depart, soc_after))

            if charge is not None:
                charger, charge_window, charged_soc = charge
                charges.append(
                    PlannedCharge(
                        charger.charger_id,
                        bid,
                        charge_window,
                        self._soc_at(self.buses[bid], events[bid], charge_window.start),
                        charged_soc,
                        charger.fast,
                    )
                )
                blocked_bus[bid].append(charge_window)
                blocked_charger.setdefault(charger.charger_id, []).append(charge_window)
                events[bid].append((charge_window.end, charged_soc))

            blocked_bus[bid].append(Window(outbound.start, return_window.end))
            events[bid].append((return_window.end, soc_after))
            remaining -= load
            accessible -= acc

        final_arrival = max(t.outbound.end for t in trips)
        if final_arrival > req.latest_arrival:
            plan = EVShuttlePlan(req.request_id, False, tuple(trips), tuple(charges), final_arrival, "late")
            object.__setattr__(plan, "_bookings", ())
            return plan

        plan = EVShuttlePlan(
            req.request_id,
            True,
            tuple(trips),
            tuple(charges),
            final_arrival,
            generation=self.ledger.generation,
        )
        object.__setattr__(plan, "_bookings", tuple(self._bookings(req.request_id, trips, charges)))
        return plan

    def _bookings(
        self,
        request_id: str,
        trips: Iterable[EVShuttleTrip],
        charges: Iterable[PlannedCharge],
    ) -> Iterable[tuple[str, Window, EVShuttleUse]]:
        bookings: list[tuple[str, Window, EVShuttleUse]] = []
        for charge in charges:
            payload = EVShuttleUse(
                "charge", request_id, charge.bus_id, charge.soc_before, charge.soc_after, charge.charger_id
            )
            bookings.append((self._bus_key(charge.bus_id), charge.window, payload))
            bookings.append((self._charger_key(charge.charger_id), charge.window, payload))
        for trip in trips:
            window = Window(trip.outbound.start, trip.return_window.end)
            payload = EVShuttleUse("trip", request_id, trip.bus_id, trip.soc_depart, trip.soc_after)
            bookings.append((self._bus_key(trip.bus_id), window, payload))
        return bookings

    def commit(self, plan: EVShuttlePlan, expected_generation: int | None = None) -> Decision:
        """Atomically lock every bus trip window and charger window.

        Either all bookings land or none do; a bus or charger grabbed by a
        competing dispatch first fails the whole plan with ``resource-conflict``.
        """
        if not plan.feasible:
            return Decision(False, "infeasible", plan.reason)
        previous = self.plans.get(plan.request_id)
        if previous is not None:
            if previous is plan or (previous.trips == plan.trips and previous.charges == plan.charges):
                return Decision(True, "ok", "shuttle plan already committed", {"arrival": previous.final_arrival})
            return Decision(False, "duplicate", "request already committed to another plan")
        bookings: tuple[tuple[str, Window, EVShuttleUse], ...] | None = getattr(plan, "_bookings", None)
        if bookings is None:
            return Decision(False, "foreign-plan", "plan did not originate from this planner")
        generation = plan.generation if expected_generation is None else expected_generation
        try:
            reservations = self.ledger.bulk_reserve(bookings, expected_generation=generation)
        except StaleGeneration:
            return Decision(False, "stale", "fleet or charger calendar changed since plan; replan")
        except ConflictError as exc:
            return Decision(False, "resource-conflict", str(exc), {"request": plan.request_id})
        self.plans[plan.request_id] = plan
        return Decision(
            True,
            "ok",
            "electric shuttle plan committed",
            {
                "arrival": plan.final_arrival,
                "generation": self.ledger.generation,
                "trips": len(plan.trips),
                "charges": len(plan.charges),
                "tokens": sorted(r.token for r in reservations),
            },
        )

    def soc_at(self, bus_id: str, at: datetime) -> float:
        """Projected state of charge after every committed booking ending at or before ``at``."""
        bus = self.buses[bus_id]
        _, events = self._committed(bus_id)
        return self._soc_at(bus, events, utc(at))

    def bus_schedule(self, bus_id: str) -> tuple[Reservation[EVShuttleUse], ...]:
        return self.ledger.reservations(self._bus_key(bus_id))

    def charger_schedule(self, charger_id: str) -> tuple[Reservation[EVShuttleUse], ...]:
        return self.ledger.reservations(self._charger_key(charger_id))
