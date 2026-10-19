"""Gateway fallback/return decisions through the real EngineHarness switcher."""

from unittest.mock import Mock

import pytest

from claude_swap import gateway
from claude_swap.autoswitch import AutoSwitchEngine, ConfigWarningEvent, NoSwitchEvent, SwitchEvent, TickOutcome
from claude_swap.usage_store import FetchRecord, UsageEntry
from tests.test_autoswitch import EngineHarness, _entry_for, _usage
from tests.test_gateway_accounts import KEY


def fleet(temp_home, monkeypatch, *, strategy="best", enabled=True):
    monkeypatch.setattr(gateway, "executable", lambda: "/fake/bin/cswap")
    monkeypatch.setattr(gateway, "JITTER_FRAC", 0)
    h = EngineHarness(temp_home, include_gateway_accounts=enabled, strategy=strategy)
    h.seed(1, "one@example.com")
    h.seed(2, "two@example.com")
    h.make_live("one@example.com", 1)
    h.switcher.add_account_from_token(KEY, base_url="https://gateway.example", slot=3)
    monkeypatch.setattr(h.engine, "_freshen_target", lambda *a: "ok")
    return h


def readings(h, first=100, second=100, status="ok", **extra):
    return {
        "1": _entry_for(_usage(first), h.clock()),
        "2": _entry_for(_usage(second), h.clock()),
        "3": UsageEntry(
            sentinel="gateway",
            gateway=dict(probeStatus=status, probedAt=h.clock(), budget=None, **extra),
        ),
    }


class TestAutoswitchGateway:
    def test_mixed_gateway_failures_share_departure_grace(self, temp_home, monkeypatch):
        from tests.test_gateway_probe import respond

        h = fleet(temp_home, monkeypatch)
        h.switcher.switch_to("3", json_output=True)
        record = gateway.account_fields("https://gateway.example")
        verdict = None
        for index in range(12):
            respond(monkeypatch, 500 if index % 2 == 0 else 503)
            verdict = gateway.probe(record, KEY, now=h.clock(), previous=verdict)
            entries = readings(h, second=75)
            entries["3"] = UsageEntry(sentinel="gateway", gateway=verdict)
            outcome = h.tick_with_entries(entries)
            if outcome == TickOutcome.SWITCHED:
                break
            h.clock.advance(120)
        assert outcome == TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_gateway_hold_inside_hysteresis_is_not_all_exhausted(self, temp_home, monkeypatch):
        h = fleet(temp_home, monkeypatch)
        h.switcher.switch_to("3", json_output=True)
        assert h.tick_with_entries(readings(h, second=85)) == TickOutcome.NO_ACTION
        assert [e.reason for e in h.events if isinstance(e, NoSwitchEvent)] == ["gateway-hold"]

    def test_persistent_transient_gateway_leaves_for_oauth_after_grace(self, temp_home, monkeypatch):
        h = fleet(temp_home, monkeypatch)
        h.switcher.switch_to("3", json_output=True)
        since = h.clock()
        assert h.tick_with_entries(readings(h, second=75, status="transient", transientSince=since)) == TickOutcome.NO_ACTION
        h.clock.advance(601)
        assert h.tick_with_entries(readings(h, second=75, status="transient", transientSince=since)) == TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert [e.trigger for e in h.events if isinstance(e, SwitchEvent)] == ["gateway-transient"]

    @pytest.mark.parametrize("strategy", ["best", "consume-first"])
    def test_gateway_fallback_after_live_oauth_candidate_skipped(
        self, temp_home, monkeypatch, strategy
    ):
        h = fleet(temp_home, monkeypatch, strategy=strategy)
        monkeypatch.setattr(
            h.engine, "_freshen_target",
            AutoSwitchEngine._freshen_target.__get__(h.engine),
        )
        monkeypatch.setattr(h.switcher, "live_session_pids_for", lambda n, email: [123] if n == "2" else [])
        assert h.tick_with_entries(readings(h, second=10)) == TickOutcome.SWITCHED
        assert h.active_number() == 3

    @pytest.mark.parametrize("strategy", ["best", "consume-first"])
    @pytest.mark.parametrize("second", [10, 85])
    def test_gateway_only_after_qualifying_oauth(
        self, temp_home, monkeypatch, strategy, second
    ):
        h = fleet(temp_home, monkeypatch, strategy=strategy)
        assert h.tick_with_entries(readings(h, second=second)) == TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.make_live("one@example.com", 1)
        assert h.tick_with_entries(readings(h)) == TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_opt_in_and_no_consume_first_nudge(self, temp_home, monkeypatch):
        h = fleet(temp_home, monkeypatch, enabled=False)
        h.tick_with_entries(readings(h))
        assert h.active_number() != 3
        h = fleet(temp_home, monkeypatch, strategy="consume-first")
        h.tick_with_entries(readings(h, first=20))
        assert h.active_number() != 3

    @pytest.mark.parametrize(
        "status,extra,reason",
        [
            ("exhausted", {}, "gateway-exhausted"),
            ("unauthorized", {}, "gateway-unauthorized"),
            ("cooling", {"coolingSince": 999000}, "gateway-cooling"),
        ],
    )
    def test_departure_reasons(self, temp_home, monkeypatch, status, extra, reason):
        h = fleet(temp_home, monkeypatch)
        h.switcher.switch_to("3", json_output=True)
        result = h.tick_with_entries(readings(h, second=50, status=status, **extra))
        assert result == TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert [e.trigger for e in h.events if isinstance(e, SwitchEvent)] == [reason]

    def test_failed_gateway_return_requires_margin_but_ignores_cooldown(
        self, temp_home, monkeypatch
    ):
        h = fleet(temp_home, monkeypatch)
        h.switcher.switch_to("3", json_output=True)
        h.engine._mutate_state(lambda state: state.update(lastSwitchAt=h.clock()))
        assert (
            h.tick_with_entries(readings(h, second=85, status="exhausted"))
            == TickOutcome.BLOCKED
        )
        assert h.active_number() == 3
        assert h.tick_with_entries(readings(h, second=75, status="exhausted")) == TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_gateway_does_not_hide_oauth_model_warning(self, temp_home, monkeypatch):
        h = fleet(temp_home, monkeypatch)
        h.engine._models = ("typo-model",)
        h.engine._check_model_names(
            set(), {"1": _usage(10), "2": _usage(20), "3": "gateway"}
        )
        assert any(isinstance(event, ConfigWarningEvent) for event in h.events)

    def test_gateway_hold_and_failed_return_hysteresis_and_all_exhausted(self, temp_home, monkeypatch):
        h = fleet(temp_home, monkeypatch)
        h.switcher.switch_to("3", json_output=True)
        assert h.tick_with_entries(readings(h, second=81)) == TickOutcome.NO_ACTION
        assert h.active_number() == 3
        h.engine._mutate_state(lambda state: state.update(lastSwitchAt=h.clock()))
        assert h.tick_with_entries(readings(h, second=50)) == TickOutcome.NO_ACTION
        h.clock.advance(301)
        assert (
            h.tick_with_entries(readings(h, status="exhausted")) == TickOutcome.BLOCKED
        )
        assert h.active_number() == 3
        assert h.tick_with_entries(readings(h, second=50, status="exhausted")) == TickOutcome.SWITCHED

    def test_real_collector_probes_once_per_tick_at_gateway_cadence(
        self, temp_home, monkeypatch
    ):
        h = fleet(temp_home, monkeypatch)
        monkeypatch.setattr(
            h.switcher,
            "_run_usage_fetches",
            lambda infos, pre: {
                str(i[0]): FetchRecord(usage=_usage(100)) for i in infos
            },
        )
        probe = Mock(
            side_effect=lambda *a, **kw: dict(
                probeStatus="ok", probedAt=kw["now"], retryAt=None, budget=None
            )
        )
        monkeypatch.setattr(gateway, "probe", probe)
        assert h.engine.tick() == TickOutcome.SWITCHED
        assert probe.call_count == 1
        h.clock.advance(60)
        h.engine.tick()
        assert probe.call_count == 1
        h.clock.advance(60)
        h.engine.tick()
        assert probe.call_count == 2

    def test_budget_uses_own_threshold(self, temp_home, monkeypatch):
        h = fleet(temp_home, monkeypatch)
        h.switcher.set_account_threshold("3", 60)
        data = readings(h)
        data["3"].gateway["budget"] = {
            "pct": 65,
            "spentUsd": 325,
            "limitUsd": 500,
            "resetsAt": None,
        }
        h.tick_with_entries(data)
        assert h.active_number() != 3


@pytest.mark.parametrize("strategy", ["best", "consume-first"])
def test_gateway_recovered_oauth_waits_until_gateway_says_no(temp_home, monkeypatch, strategy):
    h = fleet(temp_home, monkeypatch, strategy=strategy)
    h.switcher.switch_to("3", json_output=True)
    assert h.tick_with_entries(readings(h, second=10)) == TickOutcome.NO_ACTION
    assert h.active_number() == 3
    assert h.tick_with_entries(readings(h, second=10, status="exhausted")) == TickOutcome.SWITCHED
    assert h.active_number() == 2
