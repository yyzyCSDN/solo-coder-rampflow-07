import threading
import unittest
from datetime import datetime, timezone, timedelta

from rampflow.ev_shuttle import (
    EVBus, Charger, ShuttleMission, EVShuttleScheduler,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 6, 1, h, m, tzinfo=UTC)


def travel(a, b):
    table = {
        ("depot", "stand-1"): 2.0, ("stand-1", "depot"): 2.0,
        ("depot", "stand-2"): 4.0, ("stand-2", "depot"): 4.0,
        ("stand-1", "stand-2"): 3.0, ("stand-2", "stand-1"): 3.0,
        ("stand-1", "terminal"): 1.0, ("terminal", "stand-1"): 1.0,
        ("stand-2", "terminal"): 1.0, ("terminal", "stand-2"): 1.0,
        ("depot", "terminal"): 3.0, ("terminal", "depot"): 3.0,
    }
    return table.get((a, b), 0.0)


def make_bus(bus_id="B1", battery=100.0, soc=None, speed=30.0, reserve=10.0,
             back=10.0):
    # 1.0 kWh/km at 30 kph => 1 km costs 1 kWh and takes 2 min.
    return EVBus(bus_id, battery, 1.0, reserve, back,
                 avg_speed_kph=speed, aux_kwh_per_min=0.0,
                 home="depot", location="depot",
                 soc_kwh=battery if soc is None else soc)


def mission(mid="M1", legs=(2.0, 2.0), ready=10, latest=11, dwell=0):
    return ShuttleMission(
        mid, "stand-1", "terminal", "depot",
        D(ready), D(latest), legs[0], legs[1], dwell_min=dwell)


class EnergyPlanningTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0)], [], travel_km=travel)

    def test_plan_accounts_for_road_energy_and_return_reserve(self):
        plan = self.scheduler.plan(mission(legs=(2.0, 2.0)), proposal_id="p1")
        self.assertTrue(plan.feasible, plan.reason)
        # deadhead depot->stand-1 (2) + 2 + 2 = 6 kWh, reserve back = 10
        self.assertAlmostEqual(plan.segment.energy_kwh, 6.0)
        self.assertAlmostEqual(plan.return_soc_kwh, 94.0)
        self.assertIsNone(plan.charge)

    def test_route_draining_below_return_reserve_is_infeasible_without_charger(self):
        # long legs: deadhead 2 + 8 + 8 = 18 kWh; battery 20 leaves only 2,
        # below the 10 kWh return reserve, and no charger is available.
        self.scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=20.0)], [], travel_km=travel)
        plan = self.scheduler.plan(
            mission(legs=(8.0, 8.0)), proposal_id="p2")
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, "energy-shortage")

    def test_commit_writes_segment_and_bumps_generation(self):
        plan = self.scheduler.plan(mission(), proposal_id="p3")
        decision = self.scheduler.commit(plan)
        self.assertTrue(decision.accepted, decision.reason)
        self.assertEqual(decision.details["bus"], "B1")
        self.assertEqual(self.scheduler.bus_generation("B1"), 1)
        self.assertEqual(decision.details["return_soc_kwh"], 94.0)

    def test_dwell_auxiliary_draw_is_included(self):
        scheduler = EVShuttleScheduler(
            [EVBus("B1", 100.0, 1.0, 10.0, 10.0, avg_speed_kph=30.0,
                   aux_kwh_per_min=0.5, location="depot")],
            [], travel_km=travel)
        plan = scheduler.plan(mission(dwell=10), proposal_id="p4")
        self.assertTrue(plan.feasible)
        # 6 kWh road + 10*0.5 aux
        self.assertAlmostEqual(plan.segment.energy_kwh, 11.0)


class ChargingReservationTests(unittest.TestCase):
    def setUp(self):
        # 60 kW delivered: a 16 kWh top-up takes 16 min.
        self.charger = Charger("FC1", "depot", power_kw=60.0 / 0.92,
                               efficiency=0.92, fast=True)

    def test_plan_inserts_charge_session_when_battery_short(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=25.0)], [self.charger], travel_km=travel)
        # long route burns 18 kWh; 25-18=7 < 10 return reserve -> top-up.
        plan = scheduler.plan(mission(legs=(8.0, 8.0)), proposal_id="c1")
        self.assertTrue(plan.feasible, plan.reason)
        self.assertIsNotNone(plan.charge)
        # Need 3 kWh at 60 kW delivered -> 3 min.
        self.assertEqual(plan.charge.window.minutes, 3)
        self.assertEqual(plan.charge.charger_id, "FC1")
        # Latest-feasible placement: charging ends exactly at departure.
        self.assertEqual(plan.charge.window.end, plan.segment.window.start)
        # Return reserve honoured after the mission.
        self.assertGreaterEqual(plan.return_soc_kwh, 10.0 - 1e-6)

    def test_commit_reserves_charger_window(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=25.0)], [self.charger], travel_km=travel)
        plan = scheduler.plan(mission(legs=(8.0, 8.0)), proposal_id="c2")
        ok = scheduler.commit(plan)
        self.assertTrue(ok.accepted, ok.reason)
        self.assertEqual(scheduler.charger_generation("FC1"), 1)
        self.assertEqual(len(scheduler.charges), 1)

    def test_no_charging_slot_returns_charger_conflict(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=25.0)], [self.charger], travel_km=travel,
            charge_lead_hours=3)
        # Fast charger occupied across the entire allowed lead-in horizon.
        from rampflow.timeutil import Window
        scheduler._charger_windows["FC1"].append(
            Window(D(6, 59), D(10, 1)))
        scheduler._charger_generation["FC1"] = 1
        plan = scheduler.plan(mission(legs=(8.0, 8.0)), proposal_id="c3")
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, "charger-conflict")

    def test_sequential_missions_recharge_after_return_for_next_run(self):
        # Battery 25 with an 18 kWh route leaves only 7 kWh on return (<10),
        # so every mission requires a top-up at the depot fast charger.
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=25.0)], [self.charger], travel_km=travel)
        first = scheduler.plan(mission("M1", legs=(8.0, 8.0), ready=10,
                                       latest=11), proposal_id="s1")
        self.assertTrue(first.feasible)
        scheduler.commit(first)
        second = scheduler.plan(
            mission("M2", legs=(8.0, 8.0), ready=12, latest=13),
            proposal_id="s2")
        self.assertTrue(second.feasible, second.reason)
        self.assertIsNotNone(second.charge)
        # Charging happens after the bus returned, not before first mission.
        self.assertGreaterEqual(
            second.charge.window.start, first.segment.window.end)
        decision = scheduler.commit(second)
        self.assertTrue(decision.accepted, decision.reason)


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.charger = Charger("FC1", "depot", power_kw=60.0 / 0.92,
                               efficiency=0.92, fast=True)

    def test_two_plans_one_bus_only_one_commit_wins(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0)], [], travel_km=travel)
        p1 = scheduler.plan(mission("M1", ready=10, latest=11), proposal_id="x1")
        p2 = scheduler.plan(mission("M2", ready=10, latest=11), proposal_id="x2")
        d1 = scheduler.commit(p1)
        d2 = scheduler.commit(p2)
        results = sorted((d1.accepted, d2.accepted))
        self.assertEqual(results, [False, True])
        loser = d2 if d1.accepted else d1
        self.assertEqual(loser.code, "resource-conflict")
        self.assertIn("bus:B1", loser.details["resource"])

    def test_two_buses_one_charger_only_one_commit_wins(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=25.0),
             make_bus("B2", battery=100.0, soc=25.0)],
            [self.charger], travel_km=travel)
        p1 = scheduler.plan(mission("M1", legs=(8.0, 8.0), ready=10,
                                    latest=11), proposal_id="y1",
                            bus_ids=["B1"])
        p2 = scheduler.plan(mission("M2", legs=(8.0, 8.0), ready=10,
                                    latest=11), proposal_id="y2",
                            bus_ids=["B2"])
        # Both plans independently pick the same fast charger slot.
        self.assertIsNotNone(p1.charge)
        self.assertIsNotNone(p2.charge)
        decisions = []
        errors = []

        def run(plan):
            try:
                decisions.append(scheduler.commit(plan))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t1 = threading.Thread(target=run, args=(p1,))
        t2 = threading.Thread(target=run, args=(p2,))
        t1.start(); t2.start(); t1.join(); t2.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(decisions), 2)
        accepted = [d for d in decisions if d.accepted]
        rejected = [d for d in decisions if not d.accepted]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].code, "resource-conflict")
        self.assertEqual(rejected[0].details["resource"], "charger:FC1")

    def test_stale_plan_after_unrelated_commit_is_rejected(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0)], [], travel_km=travel)
        stale = scheduler.plan(mission("M1", ready=12, latest=13),
                               proposal_id="z1")
        # Another dispatcher commits a different mission on the same bus first.
        newer = scheduler.plan(mission("M2", ready=10, latest=11),
                               proposal_id="z2")
        self.assertTrue(scheduler.commit(newer).accepted)
        decision = scheduler.commit(stale)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "resource-conflict")

    def test_replan_after_conflict_succeeds_against_live_schedule(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0), make_bus("B2", battery=100.0)],
            [], travel_km=travel)
        p1 = scheduler.plan(mission("M1", ready=10, latest=11), proposal_id="r1")
        p2 = scheduler.plan(mission("M2", ready=10, latest=11), proposal_id="r2")
        self.assertTrue(scheduler.commit(p1).accepted)
        self.assertFalse(scheduler.commit(p2).accepted)
        p2b = scheduler.plan(mission("M2", ready=10, latest=11),
                             proposal_id="r2b")
        self.assertTrue(p2b.feasible)
        self.assertNotEqual(p2b.bus_id, p1.bus_id)
        self.assertTrue(scheduler.commit(p2b).accepted)


class ProposalLifecycleTests(unittest.TestCase):
    def test_infeasible_plan_cannot_commit(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0, soc=20.0)], [], travel_km=travel)
        plan = scheduler.plan(mission(legs=(8.0, 8.0)), proposal_id="i1")
        self.assertFalse(plan.feasible)
        self.assertFalse(scheduler.commit(plan).accepted)

    def test_expired_plan_is_rejected(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0)], [], travel_km=travel,
            plan_ttl_min=1)
        plan = scheduler.plan(mission(), proposal_id="e1", now=D(9))
        decision = scheduler.commit(plan, now=D(9, 2))
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "plan-expired")

    def test_commit_is_idempotent_by_proposal_id(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0)], [], travel_km=travel)
        plan = scheduler.plan(mission(), proposal_id="q1")
        first = scheduler.commit(plan)
        second = scheduler.commit(plan)
        self.assertIs(first, second)
        self.assertEqual(scheduler.bus_generation("B1"), 1)

    def test_late_mission_is_infeasible(self):
        scheduler = EVShuttleScheduler(
            [make_bus("B1", battery=100.0)], [], travel_km=travel)
        m = ShuttleMission("M", "stand-1", "terminal", "depot",
                           D(10), D(10, 1), 2.0, 2.0)
        plan = scheduler.plan(m, proposal_id="l1")
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, "late")


if __name__ == "__main__":
    unittest.main()
