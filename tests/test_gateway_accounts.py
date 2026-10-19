"""Gateway records, opt-in settings, public output, and transfer boundaries."""

import io
import json
import sys

import pytest

from claude_swap import cli, gateway
from claude_swap.exceptions import ValidationError
from claude_swap.models import Platform
from claude_swap.settings import load_settings, set_setting
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.transfer import export_accounts, import_accounts

KEY = "gateway-test-secret-DO-NOT-LEAK-9x7"


@pytest.fixture
def gateway_switcher(temp_home, monkeypatch):
    monkeypatch.setattr(gateway, "executable", lambda: "/fake/bin/cswap")
    monkeypatch.setattr(gateway, "JITTER_FRAC", 0)
    monkeypatch.setattr(
        gateway,
        "probe",
        lambda record, key, **kw: {
            "probeStatus": "ok",
            "probedAt": kw["now"],
            "retryAt": None,
            "budget": None,
        },
    )
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


class TestGatewayAccounts:
    def test_gateway_replaced_by_oauth_clears_cache_and_polls(
        self, gateway_switcher, monkeypatch
    ):
        from claude_swap.usage_store import FetchRecord

        s = gateway_switcher
        email = "same@example.com"
        s.add_account_from_token(KEY, base_url="https://gateway.example", slot=1, email=email)
        record = s._get_sequence_data()["accounts"]["1"]
        store = s._usage_store
        store.collect_gateway("1", (email, ""), record, KEY, active=False, fetch=True)
        assert store.entries({"1": (email, "")})["1"].sentinel == "gateway"
        s.remove_account("1", assume_yes=True)
        assert "1" not in store._read_rows()
        s.add_account_from_token("sk-ant-oat01-dummy", slot=1, email=email)
        usage = {"five_hour": {"pct": 10.0}, "seven_day": {"pct": 20.0}}
        polled = []

        def fetch(infos, pre):
            polled.extend(str(info[0]) for info in infos)
            return {str(info[0]): FetchRecord(usage=usage) for info in infos}

        monkeypatch.setattr(s, "_run_usage_fetches", fetch)
        entry = s.usage_entries_by_account()["1"]
        assert polled == ["1"]
        assert entry.sentinel is None and entry.gateway is None
        assert entry.decision_value() == usage

    @pytest.mark.parametrize("key", [KEY, "sk-ant-oat01-dummy", "sk-ant-api03-dummy"])
    def test_raw_record_and_defaults(self, gateway_switcher, key):
        s = gateway_switcher
        s.add_account_from_token(
            key, base_url="https://gateway.example/v1/", alias="apm"
        )
        record = s._get_sequence_data()["accounts"]["1"]
        assert record["email"] == "gateway-1@token.local"
        assert s._account_kind("1") == "gateway"
        assert record["baseUrl"] == "https://gateway.example"
        assert record["auth"] == "bearer"
        assert record["probeModel"] == "claude-haiku-4-5"
        assert record["modelDiscovery"] is False
        assert record["alias"] == "apm"
        assert s._read_account_credentials("1", record["email"]) == key

    @pytest.mark.parametrize(
        "url",
        [
            "http://gateway.example",
            "https://user:secret@host",
            "https://host/?key=secret",
            "ftp://host",
            "https://",
        ],
    )
    def test_rejects_unsafe_urls_without_echo(self, gateway_switcher, url):
        with pytest.raises(ValidationError) as exc:
            gateway_switcher.add_account_from_token(KEY, base_url=url)
        assert "secret" not in str(exc.value)
        assert not gateway_switcher._get_sequence_data()["accounts"]

    @pytest.mark.parametrize(
        "first,second", [(None, "https://gw.example"), ("https://gw.example", None)]
    )
    def test_cross_kind_collision(self, gateway_switcher, first, second):
        s = gateway_switcher
        s.add_account_from_token(KEY, email="same@example.com", base_url=first)
        with pytest.raises(ValidationError, match="already exists"):
            s.add_account_from_token(KEY, email="same@example.com", base_url=second)

    def test_gateway_setting_roundtrip(self, gateway_switcher):
        root = gateway_switcher.backup_dir
        assert load_settings(root).include_gateway_accounts is False
        set_setting(root, "autoswitch.includeGatewayAccounts", "true")
        assert load_settings(root).include_gateway_accounts is True

    def test_cli_stdin(self, gateway_switcher, monkeypatch):
        monkeypatch.setattr(cli, "ClaudeAccountSwitcher", lambda **kw: gateway_switcher)
        monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cswap",
                "add-token",
                "-",
                "--base-url",
                "http://localhost:9000",
                "--auth",
                "x-api-key",
                "--alias",
                "apm",
                "--slot",
                "4",
                "--model-discovery",
            ],
        )
        cli.main()
        r = gateway_switcher._get_sequence_data()["accounts"]["4"]
        assert r["auth"] == "x-api-key" and r["alias"] == "apm" and r["modelDiscovery"]

    def test_json_and_display_no_key_leak(self, gateway_switcher, capsys, caplog):
        s = gateway_switcher
        s.add_account_from_token(KEY, base_url="https://gateway.example")
        s.switch_to("1", json_output=True)
        row = s.list_accounts(json_output=True)["accounts"][0]
        assert row["kind"] == row["usageStatus"] == "gateway"
        assert row["gateway"]["probeStatus"] == "ok"
        assert row["gateway"]["budget"] is None
        assert s.status(json_output=True)["active"]["gateway"] == row["gateway"]
        s.list_accounts()
        s.status()
        out = capsys.readouterr().out
        assert "gateway · gateway.example · bearer" in out
        assert "gateway ok" in out
        assert KEY not in out + caplog.text + json.dumps(row)

    def test_export_import_gateway_round_trip(self, gateway_switcher, temp_home):
        s = gateway_switcher
        s.add_account_from_token(
            KEY,
            base_url="https://gateway.example",
            auth="x-api-key",
            usage_url="https://gateway.example/v1/usage",
            model_discovery=True,
        )
        s.switch_to("1", json_output=True)
        path = temp_home / "gateway-export.json"
        export_accounts(s, str(path))
        original = s._get_sequence_data()["accounts"]["1"]
        data = s._get_sequence_data()
        data["accounts"] = {}
        data["sequence"] = []
        s._write_json(s.sequence_file, data)
        import_accounts(s, str(path))
        record = s._get_sequence_data()["accounts"]["1"]
        for field in ("kind", *gateway.RECORD_FIELDS):
            assert record.get(field) == original.get(field)
        assert s._read_account_credentials("1", record["email"]) == KEY

    def test_gateway_surfaces_render(self, gateway_switcher):
        from claude_swap.menubar import _account_display_usage
        from claude_swap.tui.widgets import account_card_text

        s = gateway_switcher
        s.add_account_from_token(KEY, base_url="https://gateway.example")
        row = s.accounts_snapshot().accounts[0]
        assert "gateway · gateway.example · bearer" in account_card_text(row, 100).plain
        assert "gateway ok" in account_card_text(row, 100).plain
        assert "gateway ok" in _account_display_usage(row.usage)

    def test_hidden_helper_only_active_slot(
        self, gateway_switcher, monkeypatch, capsys
    ):
        s = gateway_switcher
        s.add_account_from_token(KEY, base_url="https://gateway.example")
        monkeypatch.setattr(cli, "ClaudeAccountSwitcher", lambda: s)
        monkeypatch.setattr(sys, "argv", ["cswap", "gateway-credential", "1"])
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert capsys.readouterr() == ("", "")
        s.switch_to("1", json_output=True)
        cli.main()
        assert capsys.readouterr() == (KEY, "")


@pytest.mark.parametrize("replacement", ["url", "kind"])
def test_forced_import_refuses_active_gateway_replacement(gateway_switcher, temp_home, replacement):
    s = gateway_switcher
    s.add_account_from_token(KEY, base_url="https://gateway.example", slot=1)
    s.switch_to("1", json_output=True)
    path = temp_home / "gateway-export.json"
    export_accounts(s, str(path))
    envelope = json.loads(path.read_text())
    row = envelope["accounts"][0]
    if replacement == "url":
        row["baseUrl"] = "https://other.example"
    else:
        row["kind"] = "oauth"
        row["credentials"] = {"claudeAiOauth": {"accessToken": "import-access", "refreshToken": "import-refresh"}}
    path.write_text(json.dumps(envelope))
    sequence = s.sequence_file.read_bytes()
    settings = s._gateway_settings_path().read_bytes()
    credentials = s._read_account_credentials("1", row["email"])
    with pytest.raises(ValidationError, match="Switch away from the active gateway"):
        import_accounts(s, str(path), force=True)
    assert s.sequence_file.read_bytes() == sequence
    assert s._gateway_settings_path().read_bytes() == settings
    assert s._read_account_credentials("1", row["email"]) == credentials
    assert s.gateway_credential("1") == KEY
