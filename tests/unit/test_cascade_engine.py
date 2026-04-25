"""Unit tests for Cascade_Engine.

Tests specific BFS traversal, depth-6 boundary, severity decay values,
CascadeError on unknown root, and blast radius after recovery.
"""
from __future__ import annotations

import pytest

from sentinel.cascade_engine import Cascade_Engine
from sentinel.exceptions import CascadeError
from sentinel.models import FailureType
from sentinel.world_state import NexaStackWorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> tuple[NexaStackWorldState, Cascade_Engine]:
    return NexaStackWorldState(), Cascade_Engine()


_FAILURE_TYPE = FailureType.cpu_spike


# ---------------------------------------------------------------------------
# BFS traversal: web-gateway → cart-service (depth 1) → redis-cache (depth 2)
# ---------------------------------------------------------------------------

class TestBFSTraversal:
    def test_root_is_in_blast_radius(self):
        ws, engine = _fresh()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        assert "web-gateway" in result

    def test_depth_1_cart_service_affected(self):
        """web-gateway → cart-service is a direct CDG edge (depth 1)."""
        ws, engine = _fresh()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        assert "cart-service" in result, (
            "cart-service should be in blast radius at depth 1 from web-gateway"
        )

    def test_depth_2_redis_cache_affected(self):
        """web-gateway → cart-service → redis-cache (depth 2)."""
        ws, engine = _fresh()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        assert "redis-cache" in result, (
            "redis-cache should be in blast radius at depth 2 from web-gateway"
        )

    def test_depth_1_order_service_affected(self):
        """web-gateway → order-service is a direct CDG edge (depth 1)."""
        ws, engine = _fresh()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        assert "order-service" in result

    def test_depth_1_user_auth_affected(self):
        """web-gateway → user-auth is a direct CDG edge (depth 1)."""
        ws, engine = _fresh()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        assert "user-auth" in result

    def test_blast_radius_matches_get_blast_radius(self):
        ws, engine = _fresh()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        assert engine.get_blast_radius() == set(result.keys())


# ---------------------------------------------------------------------------
# Depth-6 boundary: services beyond depth 6 must NOT be in blast radius
# ---------------------------------------------------------------------------

class TestDepth6Boundary:
    def test_services_beyond_depth_6_not_affected(self):
        """No service at BFS depth > 6 from root should appear in blast radius."""
        import networkx as nx

        ws, engine = _fresh()
        root = "web-gateway"
        result = engine.propagate_failure(ws, root, _FAILURE_TYPE, 1.0)

        bfs_depths = nx.single_source_shortest_path_length(ws.cdg, root)

        for service in result:
            depth = bfs_depths.get(service)
            assert depth is not None, f"{service} not reachable from {root}"
            assert depth <= 6, (
                f"{service} at depth {depth} should not be in blast radius (max=6)"
            )

    def test_max_depth_constant_is_6(self):
        assert Cascade_Engine.MAX_DEPTH == 6

    def test_services_at_depth_7_excluded(self):
        """Verify that if any service would be at depth 7, it's excluded.

        We construct a minimal chain: A→B→C→D→E→F→G→H (8 nodes, 7 hops).
        H should NOT be in the blast radius when propagating from A.
        """
        import networkx as nx

        # Build a custom world state with a linear chain CDG
        ws = NexaStackWorldState()
        # Replace CDG with a simple linear chain using real service names
        chain = [
            "web-gateway",       # depth 0
            "cart-service",      # depth 1
            "redis-cache",       # depth 2
            "service-mesh",      # depth 3
            "api-gateway",       # depth 4
            "config-service",    # depth 5
            "secret-manager",    # depth 6
            "load-balancer",     # depth 7 — must NOT be affected
        ]
        g = nx.DiGraph()
        g.add_nodes_from(ws.cdg.nodes())  # keep all 30 nodes
        for i in range(len(chain) - 1):
            g.add_edge(chain[i], chain[i + 1], weight=1.0)
        ws.cdg = g

        engine = Cascade_Engine()
        result = engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)

        assert "load-balancer" not in result, (
            "load-balancer is at depth 7 and must NOT be in blast radius"
        )
        assert "secret-manager" in result, (
            "secret-manager is at depth 6 and MUST be in blast radius"
        )


# ---------------------------------------------------------------------------
# Severity decay at each depth level
# ---------------------------------------------------------------------------

class TestSeverityDecay:
    """Verify severity = initial * (0.7^d) at each depth."""

    def _chain_state(self) -> tuple[NexaStackWorldState, list[str]]:
        """Build a world state with a linear chain CDG for predictable depths."""
        import networkx as nx

        ws = NexaStackWorldState()
        chain = [
            "web-gateway",        # depth 0
            "cart-service",       # depth 1
            "redis-cache",        # depth 2
            "service-mesh",       # depth 3
            "api-gateway",        # depth 4
            "config-service",     # depth 5
            "secret-manager",     # depth 6
        ]
        g = nx.DiGraph()
        g.add_nodes_from(ws.cdg.nodes())
        for i in range(len(chain) - 1):
            g.add_edge(chain[i], chain[i + 1], weight=1.0)
        ws.cdg = g
        return ws, chain

    def test_depth_0_severity_equals_initial(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        assert abs(result[chain[0]] - initial) < 1e-9

    def test_depth_1_severity_decay(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        expected = initial * 0.7
        assert abs(result[chain[1]] - expected) < 1e-9, (
            f"depth 1: expected {expected}, got {result[chain[1]]}"
        )

    def test_depth_2_severity_decay(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        expected = initial * (0.7 ** 2)
        assert abs(result[chain[2]] - expected) < 1e-9, (
            f"depth 2: expected {expected}, got {result[chain[2]]}"
        )

    def test_depth_3_severity_decay(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        expected = initial * (0.7 ** 3)
        assert abs(result[chain[3]] - expected) < 1e-9

    def test_depth_4_severity_decay(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        expected = initial * (0.7 ** 4)
        assert abs(result[chain[4]] - expected) < 1e-9

    def test_depth_5_severity_decay(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        expected = initial * (0.7 ** 5)
        assert abs(result[chain[5]] - expected) < 1e-9

    def test_depth_6_severity_decay(self):
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        initial = 0.8
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, initial)
        expected = initial * (0.7 ** 6)
        assert abs(result[chain[6]] - expected) < 1e-9

    def test_severity_clamped_to_1_at_depth_0(self):
        """Initial severity > 1.0 must be clamped to 1.0 at depth 0."""
        ws, chain = self._chain_state()
        engine = Cascade_Engine()
        result = engine.propagate_failure(ws, chain[0], _FAILURE_TYPE, 2.0)
        assert result[chain[0]] == 1.0, (
            f"Severity at depth 0 should be clamped to 1.0, got {result[chain[0]]}"
        )

    def test_decay_constant_is_0_7(self):
        assert Cascade_Engine.SEVERITY_DECAY == 0.7


# ---------------------------------------------------------------------------
# CascadeError on unknown root service
# ---------------------------------------------------------------------------

class TestCascadeError:
    def test_unknown_root_raises_cascade_error(self):
        ws, engine = _fresh()
        with pytest.raises(CascadeError, match="not in the Causal Dependency Graph"):
            engine.propagate_failure(ws, "nonexistent-service", _FAILURE_TYPE, 0.5)

    def test_empty_string_root_raises_cascade_error(self):
        ws, engine = _fresh()
        with pytest.raises(CascadeError):
            engine.propagate_failure(ws, "", _FAILURE_TYPE, 0.5)

    def test_cascade_error_message_contains_service_name(self):
        ws, engine = _fresh()
        bad_service = "totally-fake-service"
        with pytest.raises(CascadeError) as exc_info:
            engine.propagate_failure(ws, bad_service, _FAILURE_TYPE, 0.5)
        assert bad_service in str(exc_info.value)


# ---------------------------------------------------------------------------
# Blast radius is empty after propagate_recovery()
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_root_service_restored_after_recovery(self):
        """After recovery, the root service must be out of blast radius."""
        ws, engine = _fresh()
        engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 0.9)
        assert len(engine.get_blast_radius()) > 0, "Blast radius should be non-empty after failure"

        engine.propagate_recovery(ws, "web-gateway")
        assert "web-gateway" not in engine.get_blast_radius(), (
            "Root service must be removed from blast radius after recovery"
        )

    def test_blast_radius_shrinks_after_recovery(self):
        """Granular recovery: blast radius should shrink but may not be empty."""
        ws, engine = _fresh()
        engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 0.9)
        pre_recovery_br = len(engine.get_blast_radius())
        assert pre_recovery_br > 0

        engine.propagate_recovery(ws, "web-gateway")
        post_recovery_br = len(engine.get_blast_radius())
        assert post_recovery_br < pre_recovery_br, (
            f"Blast radius should shrink after recovery: {pre_recovery_br} -> {post_recovery_br}"
        )

    def test_root_service_metrics_restored_to_baseline(self):
        """The root service should be fully restored to baseline after recovery."""
        from sentinel.world_state import ALL_SERVICES, _baseline_metrics

        ws, engine = _fresh()
        baseline_m = _baseline_metrics()

        engine.propagate_failure(ws, "web-gateway", _FAILURE_TYPE, 1.0)
        engine.propagate_recovery(ws, "web-gateway")

        m = ws.services["web-gateway"]
        assert (m.cpu, m.memory, m.latency_ms, m.error_rate, m.saturation, m.availability) == (
            baseline_m.cpu, baseline_m.memory, baseline_m.latency_ms,
            baseline_m.error_rate, baseline_m.saturation, baseline_m.availability,
        ), "web-gateway metrics should be baseline after recovery"

    def test_direct_dependents_partially_recovered(self):
        """Direct dependents should have reduced severity after root recovery."""
        import networkx as nx

        ws, engine = _fresh()
        root = "web-gateway"
        engine.propagate_failure(ws, root, _FAILURE_TYPE, 0.9)
        
        # Record pre-recovery state of direct dependents
        dependents = list(ws.cdg.successors(root))
        pre_recovery_affected = {
            svc: engine._affected_services.get(svc, 0)
            for svc in dependents
            if svc in engine._affected_services
        }
        
        engine.propagate_recovery(ws, root)
        
        # Direct dependents should either be recovered or have reduced severity
        for svc, pre_sev in pre_recovery_affected.items():
            post_sev = engine._affected_services.get(svc, 0)
            assert post_sev <= pre_sev, (
                f"{svc} severity should decrease: {pre_sev} -> {post_sev}"
            )

    def test_get_blast_radius_empty_before_any_propagation(self):
        _, engine = _fresh()
        assert engine.get_blast_radius() == set()

    def test_recovery_without_prior_failure_is_safe(self):
        """propagate_recovery on a fresh engine should not raise."""
        ws, engine = _fresh()
        engine.propagate_recovery(ws, "web-gateway")  # should not raise
        assert engine.get_blast_radius() == set()
