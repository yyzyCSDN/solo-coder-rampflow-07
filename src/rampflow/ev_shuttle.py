"""Electric shuttle-bus mission planning with energy-aware charging reservations.

The scheduler is deliberately split into two phases:

* ``plan`` performs a side-effect-free projection of road energy, charger
  occupancy and return-to-depot reserve, returning an immutable proposal.
* ``commit`` atomically converts one proposal into a mission segment and an
  optional charger session.

Every bus and charger carries a monotonic generation counter. A proposal
snapshots the generations of every resource it touches; ``commit`` validates
those generations plus window overlap inside a single lock, so two planners
racing for the same bus or the unique fast charger cannot both win.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Iterable, Callable, Mapping

from .timeutil import Window, utc
from .models import Decision

_KIND_MISSION = "mission"
_KIND_CHARGE = "charge"
_EPS = 1e-6


@dataclass(frozen=True)
class EVBus:
    bus_id: str
    battery_kwh: float
    consumption_kwh_per_km: float
    reserve_kwh: float
    return_reserve_kwh: float
    avg_speed_kph: float = 0.0
    aux_kwh_per_min: float = 0.0
    home: str = "depot"
    location: str = "depot"
    soc_kwh: float | None = None

    def __post_init__(self):
        if self.battery_kwh <= 0 or self.consumption_kwh_per_km < 0:
            raise ValueError("invalid battery or consumption")
        if self.reserve_kwh < 0 or self.return_reserve_kwh < 0:
            raise ValueError("reserves must be non-negative")
        if self.reserve_kwh + self.return_reserve_kwh > self.battery_kwh + _EPS:
            raise ValueError("reserves exceed battery capacity")
        if self.avg_speed_kph < 0:
            raise ValueError("invalid speed")
        if self.soc_kwh is None:
            object.__setattr__(self, "soc_kwh", self.battery_kwh)
        elif not 0 <= self.soc_kwh <= self.battery_kwh + _EPS:
            raise ValueError("soc outside battery capacity")

    @property
    def min_dispatch_kwh(self) -> float:
        return self.reserve_kwh + self.return_reserve_kwh

    def travel_min(self, km: float) -> int:
        if self.avg_speed_kph <= _EPS or km <= _EPS:
            return 0
        return int((km / self.avg_speed_kph * 60) + 0.9999)


@dataclass(frozen=True)
class Charger:
    charger_id: str
    location: str
    power_kw: float
    efficiency: float = 0.92
    fast: bool = True

    def __post_init__(self):
        if self.power_kw <= 0 or not 0 < self.efficiency <= 1:
            raise ValueError("invalid charger")

    @property
    def delivered_kw(self) -> float:
        return self.power_kw * self.efficiency


@dataclass(frozen=True)
class ShuttleMission:
    mission_id: str
    pickup_location: str
    dropoff_location: str
    return_location: str
    pickup_after: datetime
    latest_dropoff: datetime
    pickup_to_dropoff_km: float
    dropoff_to_return_km: float
    dwell_min: int = 0
    require_return: bool = True

    def __post_init__(self):
        object.__setattr__(self, "pickup_after", utc(self.pickup_after))
        object.__setattr__(self, "latest_dropoff", utc(self.latest_dropoff))
        if self.pickup_to_dropoff_km < 0 or self.dropoff_to_return_km < 0:
            raise ValueError("leg distances must be non-negative")
        if self.latest_dropoff <= self.pickup_after or self.dwell_min < 0:
            raise ValueError("invalid mission timing")


@dataclass(frozen=True)
class _TimelineEntry:
    window: Window
    kind: str
    mission_id: str
    battery_after: float
    location_after: str
    session: "ChargeSession | None" = None


@dataclass(frozen=True)
class ChargeSession:
    mission_id: str
    bus_id: str
    charger_id: str
    window: Window
    location: str
    energy_delivered_kwh: float
    battery_before: float
    battery_after: float
    fast: bool


@dataclass(frozen=True)
class MissionSegment:
    mission_id: str
    bus_id: str
    window: Window
    start_location: str
    pickup_location: str
    dropoff_location: str
    return_location: str
    deadhead_km: float
    energy_kwh: float
    battery_before: float
    battery_after: float
    require_return: bool = True


@dataclass(frozen=True)
class EVDispatchPlan:
    proposal_id: str
    mission_id: str
    feasible: bool
    bus_id: str = ""
    segment: MissionSegment | None = None
    charge: ChargeSession | None = None
    return_soc_kwh: float = 0.0
    resource_generations: Mapping[str, int] = field(default_factory=dict)
    planned_revision: int = 0
    expires_at: datetime | None = None
    reason: str = ""
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", utc(self.expires_at))


class EVShuttleScheduler:
    def __init__(self, buses: Iterable[EVBus], chargers: Iterable[Charger],
                 travel_km: Callable[[str, str], float] | None = None,
                 plan_ttl_min: int = 5, charge_lead_hours: int = 24):
        self.buses = {b.bus_id: b for b in buses}
        self.chargers = {c.charger_id: c for c in chargers}
        if not self.buses:
            raise ValueError("at least one bus required")
        self.travel_km = travel_km or (lambda a, b: 0.0)
        self.plan_ttl_min = plan_ttl_min
        self.charge_lead_hours = charge_lead_hours
        self._timeline: dict[str, list[_TimelineEntry]] = {
            b: [] for b in self.buses}
        self._charger_windows: dict[str, list[Window]] = {
            c: [] for c in self.chargers}
        self._bus_generation = {b: 0 for b in self.buses}
        self._charger_generation = {c: 0 for c in self.chargers}
        self._revision = 0
        self.segments: dict[str, MissionSegment] = {}
        self.charges: list[ChargeSession] = []
        self._committed: dict[str, Decision] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------ utils
    def _distance(self, a: str, b: str) -> float:
        return 0.0 if a == b else max(0.0, float(self.travel_km(a, b)))

    def _free_after(self, windows: list[Window], at: datetime) -> datetime:
        cur = at
        changed = True
        while changed:
            changed = False
            for w in sorted(windows, key=lambda x: x.start):
                if w.start <= cur < w.end:
                    cur = w.end
                    changed = True
        return cur

    def _state_at(self, bus_id: str, at: datetime) -> tuple[float, str]:
        """Battery/location state of a bus just before instant ``at``."""
        bus = self.buses[bus_id]
        battery, location = bus.soc_kwh, bus.location
        for entry in sorted(self._timeline[bus_id], key=lambda e: e.window.start):
            if entry.window.start >= at:
                break
            battery, location = entry.battery_after, entry.location_after
        return battery, location

    def _road_energy(self, bus: EVBus, km: float) -> float:
        return km * bus.consumption_kwh_per_km

    def _mission_window(self, bus: EVBus, mission: ShuttleMission,
                        deadhead_km: float, depart: datetime) -> Window:
        minutes = (bus.travel_min(deadhead_km)
                   + bus.travel_min(mission.pickup_to_dropoff_km)
                   + mission.dwell_min
                   + bus.travel_min(mission.dropoff_to_return_km))
        return Window(depart, depart + timedelta(minutes=minutes))

    def _mission_energy(self, bus: EVBus, mission: ShuttleMission,
                        deadhead_km: float) -> float:
        road = self._road_energy(bus, deadhead_km
                                 + mission.pickup_to_dropoff_km
                                 + mission.dropoff_to_return_km)
        return road + mission.dwell_min * bus.aux_kwh_per_min

    # ------------------------------------------------------------- projection
    def _try_charge(self, bus: EVBus, mission: ShuttleMission,
                    location: str, depart: datetime, battery_at_depart: float,
                    road_energy: float, reserve_need: float):
        """Latest-feasible, shortest charging session before departure.

        A bus can only charge where it already stands between commitments
        (missions return buses to the depot). Pushing the session as late as
        possible keeps the (typically unique) fast charger open for everyone
        else earlier in the day. Returns (session, battery_at_depart).
        """
        need = road_energy + reserve_need - battery_at_depart
        if need <= _EPS:
            return None
        deadline = depart
        lower = mission.pickup_after - timedelta(hours=self.charge_lead_hours)
        blocked = [e.window for e in self._timeline[bus.bus_id]]
        best = None
        for charger in self.chargers.values():
            if charger.location != location:
                continue
            minutes = int(need / charger.delivered_kw * 60 + 0.9999)
            found = None
            window_start = deadline - timedelta(minutes=minutes)
            while window_start >= lower:
                candidate = Window(window_start,
                                   window_start + timedelta(minutes=minutes))
                hits = [w for w in blocked
                        if w.overlaps(candidate)]
                hits += [w for w in self._charger_windows[charger.charger_id]
                         if w.overlaps(candidate)]
                if not hits:
                    found = candidate
                    break
                window_start = min(h.start for h in hits) - timedelta(minutes=minutes)
            if found is None:
                continue
            energy_gain = charger.delivered_kw * minutes / 60.0
            after = min(bus.battery_kwh, battery_at_depart + energy_gain)
            delivered = after - battery_at_depart
            if after - road_energy < reserve_need - _EPS:
                continue
            session = ChargeSession(
                mission.mission_id, bus.bus_id, charger.charger_id,
                found, location, delivered,
                battery_at_depart, after, charger.fast)
            key = (minutes, -found.start.timestamp(),
                   0 if charger.fast else 1, charger.charger_id)
            if best is None or key < best[0]:
                best = (key, session, after)
        return None if best is None else (best[1], best[2])

    def _project_bus(self, bus: EVBus, mission: ShuttleMission):
        depart = self._free_after(
            [e.window for e in self._timeline[bus.bus_id]],
            mission.pickup_after)
        battery, location = self._state_at(bus.bus_id, depart)
        deadhead_km = self._distance(location, mission.pickup_location)
        window = self._mission_window(bus, mission, deadhead_km, depart)
        if any(e.window.overlaps(window)
               for e in self._timeline[bus.bus_id]):
            return None, "bus-conflict"
        if window.end > mission.latest_dropoff:
            return None, "late"

        road_energy = self._mission_energy(bus, mission, deadhead_km)
        # At return the battery must still clear the hard floor plus the
        # return-to-depot reserve the dispatcher wants to keep in hand.
        reserve_need = (bus.return_reserve_kwh if mission.require_return
                        else bus.reserve_kwh)

        charge = None
        battery_at_depart = battery
        if road_energy + reserve_need > battery + _EPS:
            charge_attempt = self._try_charge(
                bus, mission, location, depart, battery,
                road_energy, reserve_need)
            if charge_attempt is None:
                return None, ("charger-conflict" if self.chargers
                              else "energy-shortage")
            charge, battery_at_depart = charge_attempt
            if any(e.window.overlaps(charge.window)
                   for e in self._timeline[bus.bus_id]):
                return None, "bus-conflict"

        battery_after = battery_at_depart - road_energy
        if battery_after < reserve_need - _EPS:
            return None, "energy-shortage"

        segment = MissionSegment(
            mission.mission_id, bus.bus_id, window,
            location, mission.pickup_location, mission.dropoff_location,
            mission.return_location, deadhead_km, road_energy,
            battery_at_depart, battery_after, mission.require_return)
        score = (0 if charge is None else 1,
                 charge.energy_delivered_kwh if charge else 0.0,
                 window.end, bus.bus_id)
        return (score, segment, charge, battery_after), ""

    # -------------------------------------------------------------------- plan
    def plan(self, mission: ShuttleMission, *, proposal_id: str,
             now: datetime | None = None,
             bus_ids: Iterable[str] | None = None) -> EVDispatchPlan:
        now = utc(now) if now else mission.pickup_after
        pool = [self.buses[b] for b in bus_ids] if bus_ids is not None \
            else list(self.buses.values())
        candidates = []
        reasons: set[str] = set()
        for bus in pool:
            projection, reason = self._project_bus(bus, mission)
            if projection is None:
                reasons.add(reason)
            else:
                candidates.append(projection)

        if not candidates:
            reason = ("charger-conflict" if "charger-conflict" in reasons else
                      "energy-shortage" if "energy-shortage" in reasons else
                      "late" if reasons == {"late"} else "no-bus-available")
            return EVDispatchPlan(proposal_id, mission.mission_id, False,
                                  reason=reason, reasons=tuple(sorted(reasons)))

        candidates.sort(key=lambda p: p[0])
        _, segment, charge, return_kwh = candidates[0]
        tokens = {f"bus:{segment.bus_id}": self._bus_generation[segment.bus_id]}
        if charge is not None:
            tokens[f"charger:{charge.charger_id}"] = \
                self._charger_generation[charge.charger_id]
        return EVDispatchPlan(
            proposal_id, mission.mission_id, True,
            bus_id=segment.bus_id, segment=segment, charge=charge,
            return_soc_kwh=return_kwh,
            resource_generations=tokens,
            planned_revision=self._revision,
            expires_at=now + timedelta(minutes=self.plan_ttl_min))

    # ------------------------------------------------------------------ commit
    def commit(self, plan: EVDispatchPlan, *, now: datetime | None = None) -> Decision:
        if not plan.feasible or plan.segment is None:
            return Decision(False, "infeasible", plan.reason or "plan not feasible")
        with self._lock:
            prior = self._committed.get(plan.proposal_id)
            if prior is not None:
                return prior
            at = utc(now) if now else plan.segment.window.start
            if plan.expires_at is not None and at > plan.expires_at:
                return Decision(False, "plan-expired",
                                "proposal expired; replan required")

            seg = plan.segment
            bus_id = seg.bus_id
            if bus_id not in self.buses:
                return Decision(False, "unknown-bus", f"bus {bus_id} not found")
            conflict = self._validate(plan, seg, at)
            if conflict is not None:
                return conflict

            charge = plan.charge
            if charge is not None:
                self._charger_windows[charge.charger_id].append(charge.window)
                self._timeline[bus_id].append(_TimelineEntry(
                    charge.window, _KIND_CHARGE, charge.mission_id,
                    charge.battery_after, charge.location, charge))
                self.charges.append(charge)
                self._charger_generation[charge.charger_id] += 1
            self._timeline[bus_id].append(_TimelineEntry(
                seg.window, _KIND_MISSION, seg.mission_id,
                seg.battery_after, seg.return_location))
            self.segments[seg.mission_id] = seg
            self._bus_generation[bus_id] += 1
            self._revision += 1
            decision = Decision(True, "ok", "EV shuttle plan committed", {
                "mission": seg.mission_id,
                "bus": bus_id,
                "charger": charge.charger_id if charge else "",
                "charged_kwh": round(charge.energy_delivered_kwh, 3)
                if charge else 0.0,
                "return_soc_kwh": round(seg.battery_after, 3),
                "revision": self._revision,
            })
            self._committed[plan.proposal_id] = decision
            return decision

    def _validate(self, plan: EVDispatchPlan, seg: MissionSegment,
                  at: datetime) -> Decision | None:
        bus_id = seg.bus_id
        if plan.resource_generations.get(f"bus:{bus_id}") != \
                self._bus_generation[bus_id]:
            return Decision(False, "resource-conflict",
                            "bus schedule changed since planning",
                            {"resource": f"bus:{bus_id}"})
        if any(e.window.overlaps(seg.window) for e in self._timeline[bus_id]):
            return Decision(False, "resource-conflict", "bus window taken",
                            {"resource": f"bus:{bus_id}"})

        charge = plan.charge
        if charge is not None:
            if charge.charger_id not in self.chargers:
                return Decision(False, "unknown-charger",
                                f"charger {charge.charger_id} not found")
            if plan.resource_generations.get(f"charger:{charge.charger_id}") != \
                    self._charger_generation[charge.charger_id]:
                return Decision(False, "resource-conflict",
                                "charger schedule changed since planning",
                                {"resource": f"charger:{charge.charger_id}"})
            if any(w.overlaps(charge.window)
                   for w in self._charger_windows[charge.charger_id]):
                return Decision(False, "resource-conflict",
                                "charger window taken",
                                {"resource": f"charger:{charge.charger_id}"})
            if any(e.window.overlaps(charge.window)
                   for e in self._timeline[bus_id]):
                return Decision(False, "resource-conflict",
                                "bus busy during planned charge",
                                {"resource": f"bus:{bus_id}"})

        # Re-run the energy ledger against live state.
        live_battery, _ = self._state_at(bus_id, seg.window.start)
        start_battery = live_battery
        if charge is not None:
            if abs(charge.battery_before - live_battery) > 1e-3:
                return Decision(False, "resource-conflict",
                                "battery state changed since planning",
                                {"resource": f"bus:{bus_id}"})
            start_battery = charge.battery_after
        projected = start_battery - seg.energy_kwh
        if abs(projected - seg.battery_after) > 1e-3:
            return Decision(False, "resource-conflict",
                            "energy projection stale",
                            {"resource": f"bus:{bus_id}"})
        bus = self.buses[bus_id]
        floor = (bus.return_reserve_kwh if seg.require_return
                 else bus.reserve_kwh)
        if projected < floor - _EPS:
            return Decision(False, "reserve-violated",
                            "return-to-depot reserve would be breached")
        return None

    # ------------------------------------------------------------- inspection
    def bus_generation(self, bus_id: str) -> int:
        return self._bus_generation[bus_id]

    def charger_generation(self, charger_id: str) -> int:
        return self._charger_generation[charger_id]
