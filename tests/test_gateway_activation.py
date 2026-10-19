"""Settings switch transactions preserve both user settings and saved OAuth."""

import json
import sys

import pytest

from claude_swap import gateway
from claude_swap.exceptions import SwitchError
from claude_swap.paths import get_credentials_path, get_global_config_path
from tests.test_gateway_accounts import KEY
from tests.test_gateway_accounts import gateway_switcher as gateway_switcher


def seed_oauth(s):
    s.add_account_from_token("sk-ant-oat01-dummy", slot=1)
    s.switch_to("1", json_output=True)
    s.add_account_from_token(KEY, base_url="https://gateway.example", slot=2)


class TestGatewayActivation:
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
