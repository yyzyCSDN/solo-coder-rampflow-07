import threading
import unittest
from datetime import datetime, timezone

from rampflow.ev_shuttle import (
    Charger,
    ElectricBus,
    EVShuttlePlanner,
    EVShuttleUse,
    ShuttleMissionRequest,
)
from rampflow.models import Decision
from rampflow.workflows.control import RampWorkflowControl

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def request(rid="Q", passengers=40, ready=10, latest=11, travel=5, dwell=3, flight="F1"):
    return ShuttleMissionRequest(
        rid, flight, passengers, 0, "R12", "T1", D(ready), D(latest), travel, dwell
    )


class EVShuttleTests(unittest.TestCase):
    def test_plan_computes_road_energy_and_return_reserve_without_mutation(self):
        bus = ElectricBus("B1", seats=40, standing=10, accessible_spaces=0, soc_pct=10.0)
        fast = Charger("FC1", fast=True, pct_per_min=2.0)
        planner = EVShuttlePlanner([bus], [fast])

        plan = planner.plan(request())

        self.assertTrue(plan.feasible, plan.reason)
        self.assertEqual(len(plan.trips), 1)
        trip = plan.trips[0]
        # round trip: 10 travel + 6 dwell minutes -> 5.5 + 0.48 = 5.98 pct;
        # depart soc must cover energy plus the 20 pct return reserve.
        self.assertGreaterEqual(trip.soc_depart, 5.98 + bus.return_reserve_pct - 1e-9)
        self.assertAlmostEqual(trip.soc_depart - trip.soc_after, 5.98, places=6)
        self.assertGreaterEqual(trip.soc_after, bus.return_reserve_pct - 1e-9)
        # the charging window is part of the proposal and ends before departure.
        self.assertEqual(len(plan.charges), 1)
        self.assertTrue(plan.charges[0].window.end <= trip.outbound.start)
        # a plan locks nothing: generation still zero, no reservations.
        self.assertEqual(planner.generation, 0)
        self.assertEqual(planner.bus_schedule("B1"), ())
        self.assertEqual(planner.charger_schedule("FC1"), ())

    def test_bus_with_enough_charge_needs_no_charger(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=90.0)
        planner = EVShuttlePlanner([bus], [Charger("FC1", True, 2.0)])

        plan = planner.plan(request())

        self.assertTrue(plan.feasible, plan.reason)
        self.assertEqual(plan.charges, ())
        decision = planner.commit(plan)
        self.assertTrue(decision.accepted)
        self.assertEqual(planner.charger_schedule("FC1"), ())

    def test_slow_charger_preferred_when_it_finishes_in_time(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=10.0)
        slow = Charger("SC1", fast=False, pct_per_min=1.0)
        fast = Charger("FC1", fast=True, pct_per_min=2.0)
        planner = EVShuttlePlanner([bus], [slow, fast])

        plan = planner.plan(request(ready=10, latest=11, travel=5))

        self.assertTrue(plan.feasible, plan.reason)
        self.assertEqual(plan.charges[0].charger_id, "SC1")
        self.assertFalse(plan.charges[0].fast)

    def test_fast_charger_used_only_when_slow_cannot_finish_before_departure(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=10.0)
        slow = Charger("SC1", fast=False, pct_per_min=0.1)
        fast = Charger("FC1", fast=True, pct_per_min=2.0)
        planner = EVShuttlePlanner([bus], [slow, fast], charge_horizon_min=60)

        plan = planner.plan(request(ready=10, latest=11, travel=5))

        self.assertTrue(plan.feasible, plan.reason)
        self.assertEqual(plan.charges[0].charger_id, "FC1")

    def test_infeasible_when_no_charging_window_exists(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=10.0)
        planner = EVShuttlePlanner([bus], [])

        plan = planner.plan(request())

        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, "charger-capacity")
        # an infeasible plan cannot lock anything.
        decision = planner.commit(plan)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "infeasible")
        self.assertEqual(planner.generation, 0)

    def test_reserve_violation_blocks_dispatch_even_with_a_charger(self):
        # Reserve 90 pct and only 20 minutes of charging horizon: the bus can
        # never reach energy + reserve before departure.
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=10.0, return_reserve_pct=90.0)
        planner = EVShuttlePlanner([bus], [Charger("FC1", True, 2.0)], charge_horizon_min=20)

        plan = planner.plan(request())

        self.assertFalse(plan.feasible)
        self.assertEqual(plan.reason, "charger-capacity")
        self.assertEqual(planner.bus_schedule("B1"), ())

    def test_committed_charge_blocks_the_fast_charger_for_later_plans(self):
        bus_a = ElectricBus("BA", 40, 10, 0, soc_pct=10.0)
        bus_b = ElectricBus("BB", 40, 10, 0, soc_pct=1.0)
        fast = Charger("FC1", fast=True, pct_per_min=0.8)
        planner = EVShuttlePlanner([bus_a, bus_b], [fast], charge_horizon_min=40)

        first = planner.plan(request("QA", passengers=1))
        self.assertTrue(first.feasible)
        self.assertTrue(planner.commit(first).accepted)
        # QA's charging session runs right up to its 10:00 departure and the
        # only free sliver earlier is shorter than BB's deficit; the remaining
        # bus only frees up after a follow-up charge, so the planner reports the
        # shortfall up front instead of blocking the depot later.
        second = planner.plan(
            ShuttleMissionRequest("QB", "FB", 1, 0, "R12", "T1", D(10), D(10, 10), 5, 3)
        )
        self.assertFalse(second.feasible)
        self.assertIn(second.reason, {"charger-capacity", "late"})

    def test_two_dispatches_racing_for_one_bus_only_one_wins(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=90.0)
        planner = EVShuttlePlanner([bus])
        # Identical trip windows at 10:00 for two different flights.
        plan_a = planner.plan(request("QA", passengers=10, flight="FA"))
        plan_b = planner.plan(request("QB", passengers=10, flight="FB"))
        self.assertTrue(plan_a.feasible)
        self.assertTrue(plan_b.feasible)

        decisions = []

        def commit(plan):
            decisions.append(planner.commit(plan))

        t1 = threading.Thread(target=commit, args=(plan_a,))
        t2 = threading.Thread(target=commit, args=(plan_b,))
        t1.start(); t2.start(); t1.join(); t2.join()

        accepted = [d for d in decisions if d.accepted]
        rejected = [d for d in decisions if not d.accepted]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn(rejected[0].code, {"stale", "resource-conflict"})

    def test_two_dispatches_racing_for_one_charger_only_one_wins(self):
        bus_a = ElectricBus("BA", 40, 10, 0, soc_pct=10.0)
        bus_b = ElectricBus("BB", 40, 10, 0, soc_pct=10.0)
        fast = Charger("FC1", fast=True, pct_per_min=2.0)
        planner = EVShuttlePlanner([bus_a, bus_b], [fast], charge_horizon_min=30)
        plan_a = planner.plan(request("QA", passengers=1, flight="FA"))
        plan_b = planner.plan(request("QB", passengers=1, flight="FB"))
        self.assertTrue(plan_a.feasible)
        self.assertTrue(plan_b.feasible)
        # Both plans occupy FC1 in the same pre-departure window.
        self.assertEqual(plan_a.charges[0].charger_id, "FC1")
        self.assertEqual(plan_b.charges[0].charger_id, "FC1")

        decisions = []
        lock = threading.Lock()

        def commit(plan):
            d = planner.commit(plan)
            with lock:
                decisions.append(d)

        t1 = threading.Thread(target=commit, args=(plan_a,))
        t2 = threading.Thread(target=commit, args=(plan_b,))
        t1.start(); t2.start(); t1.join(); t2.join()

        self.assertEqual(sum(d.accepted for d in decisions), 1)
        rejected = [d for d in decisions if not d.accepted][0]
        # The whole multi-resource lock fails atomically; the loser is reported
        # as stale generation or a direct overlap depending on commit ordering.
        self.assertIn(rejected.code, {"stale", "resource-conflict"})

    def test_loser_can_replan_and_take_another_bus(self):
        bus_a = ElectricBus("BA", 40, 10, 0, soc_pct=90.0)
        bus_b = ElectricBus("BB", 40, 10, 0, soc_pct=90.0)
        planner = EVShuttlePlanner([bus_a, bus_b])
        plan_a = planner.plan(request("QA", passengers=10, flight="FA"))
        plan_b = planner.plan(request("QB", passengers=10, flight="FB"))
        self.assertEqual(plan_a.trips[0].bus_id, plan_b.trips[0].bus_id)

        self.assertTrue(planner.commit(plan_a).accepted)
        lost = planner.commit(plan_b)
        self.assertFalse(lost.accepted)

        replanned = planner.plan(request("QB", passengers=10, flight="FB"))
        self.assertTrue(replanned.feasible)
        self.assertEqual(replanned.trips[0].bus_id, "BB")
        self.assertTrue(planner.commit(replanned).accepted)

    def test_commit_is_idempotent_and_rejects_a_divergent_plan(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=90.0)
        planner = EVShuttlePlanner([bus])
        plan = planner.plan(request("QA"))
        first = planner.commit(plan)
        second = planner.commit(plan)
        self.assertTrue(first.accepted and second.accepted)

        other = planner.plan(request("QA"))
        decision = planner.commit(other)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "duplicate")

    def test_stale_generation_is_rejected(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=90.0)
        planner = EVShuttlePlanner([bus])
        plan = planner.plan(request("QA"))
        # Calendar moves after the proposal but before the lock.
        other = planner.plan(request("QZ", passengers=1, flight="FZ"))
        self.assertTrue(planner.commit(other).accepted)

        decision = planner.commit(plan)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "stale")

    def test_committed_soc_projects_charging_then_trip_consumption(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=10.0)
        planner = EVShuttlePlanner([bus], [Charger("FC1", True, 2.0)])
        plan = planner.plan(request())
        self.assertTrue(planner.commit(plan).accepted)

        charge = plan.charges[0]
        trip = plan.trips[0]
        self.assertAlmostEqual(planner.soc_at("B1", charge.window.start), 10.0, places=6)
        self.assertAlmostEqual(planner.soc_at("B1", trip.outbound.start), charge.soc_after, places=6)
        self.assertAlmostEqual(planner.soc_at("B1", trip.return_window.end), trip.soc_after, places=6)
        self.assertGreaterEqual(trip.soc_after, bus.return_reserve_pct - 1e-9)

    def test_ledger_payloads_distinguish_trip_and_charge_bookings(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=10.0)
        fast = Charger("FC1", True, 2.0)
        planner = EVShuttlePlanner([bus], [fast])
        plan = planner.plan(request())
        self.assertTrue(planner.commit(plan).accepted)

        bus_rows = planner.bus_schedule("B1")
        charger_rows = planner.charger_schedule("FC1")
        kinds = sorted({r.payload.kind for r in bus_rows})
        self.assertEqual(kinds, ["charge", "trip"])
        self.assertTrue(all(isinstance(r.payload, EVShuttleUse) for r in charger_rows))
        self.assertTrue(all(r.payload.kind == "charge" for r in charger_rows))

    def test_workflow_event_published_only_after_resources_lock(self):
        bus = ElectricBus("B1", 40, 10, 0, soc_pct=90.0)
        planner = EVShuttlePlanner([bus])
        control = RampWorkflowControl()

        receipt = control.commit_ev_shuttle_dispatch(
            "op-1", "Q", D(10), lambda: planner.commit(planner.plan(request("Q")))
        )
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(receipt.topic, "ev.shuttle.dispatch.committed")
        self.assertTrue(receipt.result.accepted)


if __name__ == "__main__":
    unittest.main()
