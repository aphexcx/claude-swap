"""Settings switch transactions preserve both user settings and saved OAuth."""

import json
import sys

import pytest

from claude_swap import gateway
from claude_swap import cli
from claude_swap.exceptions import ConfigError, SwitchError, ValidationError
from claude_swap.paths import get_credentials_path, get_global_config_path
from tests.test_gateway_accounts import KEY
from tests.test_gateway_accounts import gateway_switcher as gateway_switcher


def seed_oauth(s):
    s.add_account_from_token("sk-ant-oat01-dummy", slot=1)
    s.switch_to("1", json_output=True)
    s.add_account_from_token(KEY, base_url="https://gateway.example", slot=2)


class TestGatewayActivation:
    def test_purge_refuses_active_gateway(self, gateway_switcher, monkeypatch):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        settings = s._gateway_settings_path().read_bytes()
        sequence = s.sequence_file.read_bytes()
        monkeypatch.setattr("builtins.input", lambda *a: "y")
        with pytest.raises(ConfigError, match="Switch away from the active gateway before purging"):
            s.purge()
        assert s.sequence_file.read_bytes() == sequence
        assert s._gateway_settings_path().read_bytes() == settings
        assert s.gateway_credential("2") == KEY

    def test_gateway_helper_refuses_session_profile(self, gateway_switcher, monkeypatch, capsys):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        email = s._get_sequence_data()["accounts"]["1"]["email"]
        profile = s._session_dir("1", email)
        profile.mkdir(parents=True)
        # Simulate an older session profile still linked to default settings.
        (profile / "settings.json").symlink_to(s._gateway_settings_path())
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
        monkeypatch.setattr(cli, "ClaudeAccountSwitcher", lambda: s)
        monkeypatch.setattr(sys, "argv", ["cswap", "gateway-credential", "2"])
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        output = capsys.readouterr()
        assert output.out == ""
        assert "session profile" in output.err
        assert KEY not in output.err

    @pytest.mark.parametrize("first,second", [("claude-swap", "cswap"), ("cswap", "claude-swap")])
    def test_gateway_return_across_cli_aliases(self, gateway_switcher, monkeypatch, first, second):
        s = gateway_switcher
        seed_oauth(s)
        monkeypatch.setattr(gateway, "executable", lambda: f"/fake/bin/{first}")
        s.switch_to("2", json_output=True)
        monkeypatch.setattr(gateway, "executable", lambda: f"/fake/bin/{second}")
        result = s.switch_to("1", json_output=True)
        assert result["switched"]
        assert result["from"]["number"] == 2
        settings = json.loads(s._gateway_settings_path().read_text())
        assert "apiKeyHelper" not in settings
        assert settings["env"]["ANTHROPIC_BASE_URL"] == gateway.FIRST_PARTY_URL

    @pytest.mark.parametrize("operation", ["move", "swap", "move-occupied"])
    def test_active_gateway_renumber_refused(self, gateway_switcher, operation):
        s = gateway_switcher
        seed_oauth(s)
        s.add_account_from_token(KEY, base_url="https://gateway.example", slot=3)
        s.switch_to("2", json_output=True)
        sequence = s.sequence_file.read_bytes()
        settings = s._gateway_settings_path().read_bytes()
        with pytest.raises(ValidationError, match="active gateway.*switch away"):
            if operation == "move":
                s.move_account("2", "4")
            elif operation == "swap":
                s.swap_accounts("3", "2")
            else:
                s.move_account("3", "2")
        assert s.sequence_file.read_bytes() == sequence
        assert s._gateway_settings_path().read_bytes() == settings
        assert s.gateway_credential("2") == KEY

    def test_return_keeps_oauth_backup_rotated_while_gateway_active(
        self, gateway_switcher
    ):
        s = gateway_switcher
        seed_oauth(s)
        email = s._get_sequence_data()["accounts"]["1"]["email"]
        original = json.loads(s._read_account_credentials("1", email))
        original["claudeAiOauth"].update(
            accessToken="old-access", refreshToken="old-refresh", expiresAt=1000
        )
        s._write_account_credentials("1", email, json.dumps(original))
        s.switch_to("1", force=True, json_output=True)
        s.switch_to("2", json_output=True)
        refreshed = json.loads(json.dumps(original))
        refreshed["claudeAiOauth"].update(
            accessToken="new-access", refreshToken="new-refresh", expiresAt=2000
        )
        s._write_account_credentials("1", email, json.dumps(refreshed))

        s.switch_to("1", json_output=True)

        assert json.loads(s._read_account_credentials("1", email)) == refreshed
        assert json.loads(get_credentials_path().read_text()) == refreshed
        assert s.current_account_number() == "1"

    def test_activation_and_return_preserve_saved_login(self, gateway_switcher):
        s = gateway_switcher
        seed_oauth(s)
        credentials = get_credentials_path().read_bytes()
        login = get_global_config_path().read_bytes()
        settings = s._gateway_settings_path()
        settings.write_text('{"permissions":{"allow":["Read"]},"env":{"KEEP":"yes"}}')
        s.switch_to("2", json_output=True)
        assert get_credentials_path().read_bytes() == credentials
        assert get_global_config_path().read_bytes() == login
        assert s.current_account_number() == "2"
        assert s.gateway_credential("2") == KEY
        data = json.loads(settings.read_text())
        assert KEY not in settings.read_text()
        assert data["apiKeyHelper"] == "/fake/bin/cswap gateway-credential 2"
        assert data["env"]["KEEP"] == "yes"
        if sys.platform != "win32":
            assert settings.stat().st_mode & 0o777 == 0o600
        result = s.switch_to("1", json_output=True)
        assert result["from"]["number"] == 2 and result["to"]["number"] == 1
        data = json.loads(settings.read_text())
        assert "apiKeyHelper" not in data
        assert data["env"] == {
            "KEEP": "yes",
            "ANTHROPIC_BASE_URL": gateway.FIRST_PARTY_URL,
        }
        assert data["permissions"] == {"allow": ["Read"]}
        assert s.gateway_credential("2") is None

    @pytest.mark.parametrize("exists", [False, True])
    def test_failed_sequence_write_restores_exact_settings(
        self, gateway_switcher, monkeypatch, exists
    ):
        s = gateway_switcher
        seed_oauth(s)
        path = s._gateway_settings_path()
        before = b'{ "env" : { "KEEP" : "yes" } }\n'
        if exists:
            path.write_bytes(before)
        seq_before = s.sequence_file.read_bytes()
        original = s._write_json

        def fail(path, data):
            if path == s.sequence_file and data.get("activeAccountNumber") == 2:
                raise OSError("injected sequence write failure")
            return original(path, data)

        monkeypatch.setattr(s, "_write_json", fail)
        with pytest.raises(SwitchError, match="previous settings restored"):
            s.switch_to("2", json_output=True)
        assert path.read_bytes() == before if exists else not path.exists()
        assert s.sequence_file.read_bytes() == seq_before
        assert s.current_account_number() == "1"

    def test_incomplete_rollback_is_reported_without_exception_content(
        self, gateway_switcher, monkeypatch, caplog
    ):
        s = gateway_switcher
        seed_oauth(s)
        seq_before = s.sequence_file.read_bytes()

        def fail(*args, **kwargs):
            raise OSError(KEY)

        monkeypatch.setattr(s, "_write_json", fail)
        monkeypatch.setattr(gateway, "restore_settings", fail)
        with pytest.raises(SwitchError, match="rollback was incomplete") as error:
            s.switch_to("2", json_output=True)
        assert KEY not in str(error.value)
        assert KEY not in caplog.text
        assert s.sequence_file.read_bytes() == seq_before

    def test_failed_return_restores_gateway_settings(
        self, gateway_switcher, monkeypatch
    ):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        before = s._gateway_settings_path().read_bytes()
        sequence = s.sequence_file.read_bytes()

        def fail(*args, **kwargs):
            raise OSError("injected OAuth write failure")

        monkeypatch.setattr(s, "_write_credentials", fail)
        with pytest.raises(SwitchError, match="previous settings restored"):
            s.switch_to("1", json_output=True)
        assert s._gateway_settings_path().read_bytes() == before
        assert s.sequence_file.read_bytes() == sequence
        assert s.gateway_credential("2") == KEY

    def test_foreign_helper_refused(self, gateway_switcher):
        s = gateway_switcher
        seed_oauth(s)
        path = s._gateway_settings_path()
        before = b'{"apiKeyHelper":"some-other-provider"}'
        path.write_bytes(before)
        with pytest.raises(SwitchError, match="foreign apiKeyHelper"):
            s.switch_to("2", json_output=True)
        assert path.read_bytes() == before
        assert s.current_account_number() == "1"

    def test_last_gateway_removal_cleans_only_owned_keys(self, gateway_switcher):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        s.switch_to("1", json_output=True)
        s.remove_account("2", assume_yes=True)
        assert "ANTHROPIC_BASE_URL" not in json.loads(
            s._gateway_settings_path().read_text()
        ).get("env", {})

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink mode is POSIX-only")
    def test_symlink_settings_write_through(self, gateway_switcher, temp_home):
        s = gateway_switcher
        seed_oauth(s)
        target = temp_home / "shared-settings.json"
        target.write_text('{"custom":"preserved"}')
        path = s._gateway_settings_path()
        path.symlink_to(target)
        s.switch_to("2", json_output=True)
        assert path.is_symlink()
        assert json.loads(target.read_text())["custom"] == "preserved"
        assert (
            json.loads(target.read_text())["env"]["ANTHROPIC_BASE_URL"]
            == "https://gateway.example"
        )
