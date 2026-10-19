"""Gateway HTTP verdicts, secret redaction, and persisted independent cadence."""

import io
import json
from urllib.error import HTTPError, URLError
from unittest.mock import Mock

import pytest

from claude_swap import gateway
from claude_swap.usage_store import FetchRecord, UsageStore
from tests.test_gateway_accounts import KEY


def respond(monkeypatch, status, message="", retry=None):
    body = json.dumps({"error": {"type": "test_error", "message": message}}).encode()
    response = Mock(status=status, headers={"Retry-After": retry} if retry else {})
    response.read.return_value = body
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    opener = Mock()
    if status == 200:
        opener.open.return_value = response
    else:
        opener.open.side_effect = HTTPError(
            "https://gateway.example/v1/messages",
            status,
            "error",
            response.headers,
            io.BytesIO(body),
        )
    monkeypatch.setattr(gateway, "build_opener", lambda *args: opener)
    return opener


class TestGatewayProbe:
    @pytest.mark.parametrize("reset", ["clear_dead_token", "record"])
    def test_oauth_cache_write_clears_prior_gateway_state(self, temp_home, monkeypatch, reset):
        store = UsageStore(temp_home / "cache", clock=lambda: 1000.0)
        identity = {"1": ("same@example.com", "")}
        respond(monkeypatch, 200)
        store.collect_gateway(
            "1", identity["1"], gateway.account_fields("https://gateway.example"), KEY,
            active=False, fetch=True,
        )
        assert store.entries(identity)["1"].sentinel == "gateway"
        usage = {"five_hour": {"pct": 10.0}}
        if reset == "clear_dead_token":
            store.clear_dead_token(["1"], identity)
        else:
            store.record({"1": FetchRecord(usage=usage)}, identity)
        entry = store.entries(identity)["1"]
        assert entry.sentinel is None and entry.gateway is None
        assert entry.next_poll_at is None
        assert not any(key.startswith("gateway") for key in store._read_rows()["1"])
        if reset == "record":
            assert entry.decision_value() == usage
        else:
            assert "1" in store.reserve(["1"], identity, respect_plans=True)

    @pytest.mark.parametrize(
        "status,verdict",
        [
            (200, "ok"),
            (401, "unauthorized"),
            (403, "unauthorized"),
            (429, "exhausted"),
            (503, "cooling"),
            (529, "cooling"),
            (500, "transient"),
            (404, "transient"),
        ],
    )
    def test_http_verdicts(self, monkeypatch, status, verdict):
        respond(monkeypatch, status)
        result = gateway.probe(
            gateway.account_fields("https://gateway.example"), KEY, now=100
        )
        assert result["probeStatus"] == verdict
        if status == 429:
            assert result["retryAt"] == 3700

    @pytest.mark.parametrize("auth", ["bearer", "x-api-key"])
    def test_probe_request(self, monkeypatch, auth):
        opener = respond(monkeypatch, 200)
        gateway.probe(
            gateway.account_fields(
                "http://127.0.0.1:8000", auth, probe_model="dummy-model"
            ),
            KEY,
        )
        request = opener.open.call_args.args[0]
        assert request.full_url == "http://127.0.0.1:8000/v1/messages"
        assert request.get_method() == "POST"
        assert request.headers["Anthropic-version"] == "2023-06-01"
        assert request.headers[
            "Authorization" if auth == "bearer" else "X-api-key"
        ] == ("Bearer " + KEY if auth == "bearer" else KEY)
        assert json.loads(request.data) == {
            "model": "dummy-model",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "."}],
        }
        assert opener.open.call_args.kwargs["timeout"] == 15

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("unknown model dummy", "ok"),
            ("model not found", "ok"),
            ("invalid request", "transient"),
        ],
    )
    def test_unknown_model_proves_auth(self, monkeypatch, message, expected):
        respond(monkeypatch, 400, message)
        assert (
            gateway.probe(gateway.account_fields("https://gateway.example"), KEY)[
                "probeStatus"
            ]
            == expected
        )

    def test_retry_after_and_cooling_backoff(self, monkeypatch):
        respond(monkeypatch, 429, retry="90")
        assert (
            gateway.probe(
                gateway.account_fields("https://gateway.example"), KEY, now=100
            )["retryAt"]
            == 190
        )
        previous = {}
        for count, delay in enumerate((60, 120, 240, 480, 600, 600)):
            respond(monkeypatch, 529)
            previous = gateway.probe(
                gateway.account_fields("https://gateway.example"),
                KEY,
                now=100 + count,
                previous=previous,
            )
            assert previous["retryAt"] == 100 + count + delay
            assert previous["coolingSince"] == 100

    def test_no_key_leak_in_verdict_log_json_or_exception(self, monkeypatch, caplog):
        respond(monkeypatch, 503, KEY + " " + "x" * 300)
        record = gateway.account_fields("https://gateway.example")
        verdict = gateway.probe(record, KEY)
        assert len(verdict["message"]) <= 120
        assert (
            KEY
            not in repr(verdict)
            + json.dumps(gateway.json_fields(record, verdict))
            + caplog.text
        )
        opener = Mock()
        opener.open.side_effect = URLError(KEY)
        monkeypatch.setattr(gateway, "build_opener", lambda *args: opener)
        # An exception containing the credential is swallowed and sanitized.
        result = gateway.probe(record, KEY)
        assert result["probeStatus"] == "transient"
        assert KEY not in repr(result) + caplog.text
        with pytest.raises(ValueError) as exc:
            gateway.validate_url("https://example.com?key=" + KEY)
        assert KEY not in str(exc.value)

    def test_cadences_and_unconfigured_usage(self, monkeypatch):
        monkeypatch.setattr(gateway, "JITTER_FRAC", 0)
        assert gateway.next_probe_at({"probeStatus": "ok"}, active=True, now=0) == 120
        assert gateway.next_probe_at({"probeStatus": "ok"}, active=False, now=0) == 600
        assert (
            gateway.next_probe_at(
                {"probeStatus": "exhausted", "retryAt": 20}, active=False, now=0
            )
            == 80
        )
        assert (
            gateway.next_probe_at({"probeStatus": "cooling"}, active=False, now=0) == 60
        )
        assert (
            gateway.next_probe_at({"probeStatus": "unauthorized"}, active=True, now=0)
            is None
        )
        assert gateway.usage({}, KEY) == "unconfigured"

    def test_persistence_cadence_quarantine_and_replaced_key(
        self, temp_home, monkeypatch
    ):
        now = [1000.0]
        store = UsageStore(temp_home / "cache", clock=lambda: now[0])
        record = gateway.account_fields("https://gateway.example")
        status = ["ok"]
        probe = Mock(
            side_effect=lambda *a, **kw: dict(
                probeStatus=status[0], probedAt=kw["now"], retryAt=None, budget=None
            )
        )
        monkeypatch.setattr(gateway, "probe", probe)
        monkeypatch.setattr(gateway, "JITTER_FRAC", 0)

        def collect(key=KEY, active=False):
            return store.collect_gateway(
                "1", ("gateway@token.local", ""), record, key, active=active, fetch=True
            )

        assert collect().gateway["probeStatus"] == "ok"
        collect()
        assert probe.call_count == 1
        now[0] += 120
        collect(active=True)
        assert probe.call_count == 2
        now[0] += 120
        status[0] = "unauthorized"
        collect(active=True)
        now[0] += 10000
        collect(active=True)
        assert probe.call_count == 3
        status[0] = "ok"
        collect(KEY + "-replacement")
        assert probe.call_count == 4
        rows = store._read_rows()
        assert rows["1"]["sentinel"] == "gateway"
        assert KEY not in json.dumps(rows)
