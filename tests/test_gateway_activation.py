"""Settings switch transactions preserve both user settings and saved OAuth."""

import json
import sys

import pytest

from claude_swap import gateway
from claude_swap import cli
from claude_swap.exceptions import ConfigError, SessionError, SwitchError, ValidationError
from claude_swap.paths import get_credentials_path, get_global_config_path
from tests.test_gateway_accounts import KEY
from tests.test_gateway_accounts import gateway_switcher as gateway_switcher


def seed_oauth(s):
    s.add_account_from_token("sk-ant-oat01-dummy", slot=1)
    s.switch_to("1", json_output=True)
    s.add_account_from_token(KEY, base_url="https://gateway.example", slot=2)


class TestGatewayActivation:
    @pytest.mark.parametrize("force", [False, True])
    def test_gateway_identity_match_does_not_replace_settings_activation(self, gateway_switcher, force):
        s = gateway_switcher
        seed_oauth(s)
        email = s.account_email("1")
        live = s._read_credentials()
        s.remove_account("1", assume_yes=True)
        s.add_account_from_token(KEY, base_url="https://gateway.example", slot=3, email=email)
        assert s.current_account_number() is None
        assert s.status(json_output=True)["active"] == {"email": email, "managed": False}
        assert not any(info[4] for info in s._build_accounts_info())
        assert s.switch(json_output=True)["reason"] == "unmanaged-account"
        result = s.switch_to("3", json_output=True, force=force)
        assert result["switched"] and result["from"]["number"] is None
        assert s.current_account_number() == "3"
        assert s.gateway_credential("3") == KEY
        assert s._read_credentials() == live
        assert s.switch_to("3", json_output=True)["reason"] == "already-active"

    def test_live_add_refuses_oauth_refresh_of_inactive_gateway(self, gateway_switcher, monkeypatch):
        s = gateway_switcher
        seed_oauth(s)
        email = s.account_email("1")
        s.remove_account("1", assume_yes=True)
        s.add_account_from_token(KEY, base_url="https://gateway.example", slot=3, email=email)
        monkeypatch.setattr(s, "_reject_foreign_credential_capture", lambda creds, *args: creds)
        monkeypatch.setattr(s, "_reject_credential_drift_since_verify", lambda *args: None)
        monkeypatch.setattr(s, "_reject_identity_drift_since_verify", lambda *args: None)
        sequence = s.sequence_file.read_bytes()
        with pytest.raises(ValidationError, match="gateway"):
            s.add_account()
        assert s.sequence_file.read_bytes() == sequence
        assert s._read_account_credentials("3", email) == KEY

    def test_live_add_refuses_replacing_active_gateway(self, gateway_switcher):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        with pytest.raises(ConfigError, match="Switch away from the active gateway"):
            s.add_account(slot=2, assume_yes=True)
        assert s.gateway_credential("2") == KEY

    @pytest.mark.parametrize("operation", ["purge", "add-token", "add"])
    def test_gateway_activation_during_confirmation_refuses_deletion(
        self, gateway_switcher, monkeypatch, operation
    ):
        s = gateway_switcher
        seed_oauth(s)

        def confirm(*args):
            s.switch_to("2", json_output=True)
            return "y"

        monkeypatch.setattr("builtins.input", confirm)
        with pytest.raises(ConfigError, match="Switch away from the active gateway"):
            if operation == "purge":
                s.purge()
            elif operation == "add":
                s.add_account(slot=2)
            else:
                s.add_account_from_token(KEY, base_url="https://other.example", slot=2, email="other@example.com")
        assert s.current_account_number() == "2"
        assert s.gateway_credential("2") == KEY

    @pytest.mark.parametrize("contents", ["{broken", '{"env": []}'])
    def test_malformed_gateway_settings_raises_switch_error(self, gateway_switcher, contents):
        s = gateway_switcher
        seed_oauth(s)
        s._gateway_settings_path().write_text(contents)
        with pytest.raises(SwitchError, match="Cannot read Claude settings.json"):
            s.switch_to("2", json_output=True)
        assert s._gateway_settings_path().read_text() == contents

    @pytest.mark.parametrize("name", ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR", "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"])
    def test_gateway_activation_refuses_settings_auth_overrides(self, gateway_switcher, name):
        s = gateway_switcher
        seed_oauth(s)
        path = s._gateway_settings_path()
        path.write_text(json.dumps({"env": {name: KEY}}))
        settings = path.read_bytes()
        sequence = s.sequence_file.read_bytes()
        with pytest.raises(SwitchError, match=str(path)) as error:
            s.switch_to("2", json_output=True)
        assert KEY not in str(error.value)
        assert path.read_bytes() == settings
        assert s.sequence_file.read_bytes() == sequence

    @pytest.mark.parametrize("slot,email", [(2, None), (None, "gateway-2@token.local"), (4, "gateway-2@token.local"), (2, "other@example.com")])
    def test_active_gateway_add_token_overwrite_or_migration_refused(self, gateway_switcher, slot, email):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        sequence = s.sequence_file.read_bytes()
        settings = s._gateway_settings_path().read_bytes()
        with pytest.raises(ConfigError, match="Switch away from the active gateway"):
            s.add_account_from_token(KEY, base_url="https://other.example", slot=slot, email=email, assume_yes=True)
        assert s.sequence_file.read_bytes() == sequence
        assert s._gateway_settings_path().read_bytes() == settings
        assert s.gateway_credential("2") == KEY

    def test_remove_gateway_live_session_refusal_keeps_settings(self, gateway_switcher, monkeypatch):
        s = gateway_switcher
        seed_oauth(s)
        s.switch_to("2", json_output=True)
        s.switch_to("1", json_output=True)
        settings = s._gateway_settings_path().read_bytes()
        sequence = s.sequence_file.read_bytes()
        calls = []

        def refuse_at_delete(*args):
            calls.append(args)
            if len(calls) > 1:
                raise SessionError("live session appeared")

        monkeypatch.setattr(s, "_ensure_no_live_session", refuse_at_delete)
        with pytest.raises(SessionError, match="live session appeared"):
            s.remove_account("2", assume_yes=True)
        assert s._gateway_settings_path().read_bytes() == settings
        assert s.sequence_file.read_bytes() == sequence

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
        # Simulate an older session profile with inherited gateway settings.
        (profile / "settings.json").write_bytes(s._gateway_settings_path().read_bytes())
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


def test_gateway_return_stashes_earlier_expiry_foreign_live_credentials(gateway_switcher):
    from claude_swap import oauth

    s = gateway_switcher
    seed_oauth(s)
    email = s.account_email("1")
    original = json.loads(s.read_account_credentials("1", email))
    original["claudeAiOauth"].update(accessToken="own-access", refreshToken="own-refresh", expiresAt=2000)
    backup = json.dumps(original)
    s._write_account_credentials("1", email, backup)
    get_credentials_path().write_text(backup)
    s.switch_to("2", json_output=True)
    foreign = json.dumps({"claudeAiOauth": {
        "accessToken": "foreign-access", "refreshToken": "foreign-refresh", "expiresAt": 1000,
    }})
    get_credentials_path().write_text(foreign)
    s._probe_verdicts[s._lineage_key("1", email, oauth.credential_fingerprint(foreign))] = False
    s.switch_to("1", json_output=True)
    assert s.read_account_credentials("1", email) == backup
    assert json.loads(get_credentials_path().read_text()) == original
    stash = s.list_unclaimed_credentials()
    assert len(stash) == 1
    entry_id, metadata = next(iter(stash.items()))
    assert metadata["reason"] == "known-foreign"
    assert s._store._read_unclaimed_credential(entry_id) == (foreign, False)


@pytest.mark.parametrize("missing", [False, True])
def test_gateway_entry_can_recover_without_live_oauth_credentials(gateway_switcher, missing):
    s = gateway_switcher
    seed_oauth(s)
    backup = s.read_account_credentials("1", s.account_email("1"))
    if missing:
        get_credentials_path().unlink()
    else:
        get_credentials_path().write_text("")
    s.switch_to("2", json_output=True)
    assert s.current_account_number() == "2"
    assert s.gateway_credential("2") == KEY
    assert s.read_account_credentials("1", s.account_email("1")) == backup


@pytest.mark.parametrize("live_expiry", [1000, 3000])
def test_gateway_return_resolves_foreign_live_owner_without_cached_verdict(gateway_switcher, monkeypatch, live_expiry):
    from claude_swap import oauth
    s = gateway_switcher
    seed_oauth(s)
    email = s.account_email("1")
    backup = json.dumps({"claudeAiOauth": {"accessToken": "own-access", "refreshToken": "own-refresh", "expiresAt": 2000}})
    s._write_account_credentials("1", email, backup)
    get_credentials_path().write_text(backup)
    s.switch_to("2", json_output=True)
    foreign = json.dumps({"claudeAiOauth": {"accessToken": "foreign-access", "refreshToken": "foreign-refresh", "expiresAt": live_expiry}})
    get_credentials_path().write_text(foreign)
    s._probe_verdicts.clear()
    monkeypatch.setattr(oauth, "fetch_oauth_profile", lambda *a: {"uuid": "foreign-uuid", "email": "foreign@example.com", "organizationUuid": "foreign-org"})
    s.switch_to("1", json_output=True)
    assert s.read_account_credentials("1", email) == backup
    assert any(s._store._read_unclaimed_credential(n)[0] == foreign for n in s.list_unclaimed_credentials())


@pytest.mark.parametrize("missing", [False, True])
def test_gateway_return_restores_backup_after_live_login_missing(gateway_switcher, missing):
    s = gateway_switcher
    seed_oauth(s)
    backup = s.read_account_credentials("1", s.account_email("1"))
    if missing:
        get_credentials_path().unlink()
    else:
        get_credentials_path().write_text("")
    s.switch_to("2", json_output=True)
    s.switch_to("1", json_output=True)
    assert s.current_account_number() == "1"
    assert s.read_account_credentials("1", s.account_email("1")) == backup
    assert json.loads(get_credentials_path().read_text()) == json.loads(backup)


def test_gateway_remove_rechecks_replacement_activated_during_confirmation(gateway_switcher, monkeypatch):
    s = gateway_switcher
    seed_oauth(s)
    def confirm(*args):
        s.add_account_from_token(KEY, base_url="https://replacement.example", slot=2, assume_yes=True)
        s.switch_to("2", json_output=True)
        return "y"
    monkeypatch.setattr("builtins.input", confirm)
    with pytest.raises(ConfigError, match="Switch away from the active gateway"):
        s.remove_account("2")
    assert s.current_account_number() == "2"
    assert s.gateway_credential("2") == KEY


def test_gateway_remove_holds_roster_lock_through_sequence_commit(gateway_switcher, monkeypatch):
    from claude_swap.locking import FileLock
    s = gateway_switcher
    seed_oauth(s)
    write = s._write_json
    held = []
    def checked_write(path, data):
        if path == s.sequence_file and "2" not in data.get("accounts", {}):
            lock = FileLock(s.lock_file)
            acquired = lock.acquire(timeout=0)
            held.append(not acquired)
            if acquired:
                lock.release()
        return write(path, data)
    monkeypatch.setattr(s, "_write_json", checked_write)
    s.remove_account("2", assume_yes=True)
    assert held == [True]
