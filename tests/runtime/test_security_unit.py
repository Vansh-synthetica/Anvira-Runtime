"""Runtime: secrets, app registry, permissions, redaction."""
import json
import os
import stat
import sys

import pytest

from anvira_runtime.config.paths import resolve_layout
from anvira_runtime.security.secrets import (DEFAULT_APP_PERMISSIONS, PERMISSIONS, AppRegistry, AuthError,
                                              SecretStore, redact, redact_obj)


@pytest.fixture()
def reg(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure()
    s = SecretStore(lay)
    return lay, s, AppRegistry(lay, s)


def test_secrets_generated_once_and_stable(reg, tmp_path):
    lay, s, _ = reg
    first = s.get("owner_token")
    assert len(first) == 64 and first != s.get("register_token") != s.get("orcha_token")
    assert SecretStore(lay).get("owner_token") == first   # persisted, not regenerated
    assert (lay.state_dir / "owner.token").read_text() == first
    assert (lay.state_dir / "register.token").read_text() == s.get("register_token")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_secret_files_owner_only(reg):
    lay, _, _ = reg
    for f in (lay.secrets_file, lay.state_dir / "owner.token"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_app_registration_and_default_permissions(reg):
    _, _, apps = reg
    rec, token = apps.register("anvira-notes", "Anvira Notes", ["models.manage", "chat"])
    assert token.startswith("anv_") and "token_hash" not in rec
    assert set(rec["permissions"]) == set(DEFAULT_APP_PERMISSIONS)
    assert rec["requested"] == ["models.manage"]          # requested but NOT granted
    pr = apps.authenticate(token)
    assert pr.kind == "app" and pr.app_id == "anvira-notes" and pr.has("chat") and not pr.has("models.manage")
    assert pr.namespace == "app:anvira-notes"


def test_only_hash_is_stored(reg):
    lay, _, apps = reg
    _, token = apps.register("study", None)
    assert token not in lay.apps_file.read_text()


def test_owner_token_is_all_powerful_and_apps_are_not(reg):
    _, s, apps = reg
    owner = apps.authenticate(s.get("owner_token"))
    assert owner.kind == "owner" and all(owner.has(p) for p in PERMISSIONS)
    _, token = apps.register("dev")
    assert not apps.authenticate(token).has("apps.admin")
    assert apps.authenticate("nonsense") is None and apps.authenticate("") is None


def test_grant_deny_revoke(reg):
    _, _, apps = reg
    _, token = apps.register("dev", requested=["models.manage"])
    apps.grant("dev", ["models.manage"])
    assert apps.authenticate(token).has("models.manage") and apps.get("dev")["requested"] == []
    apps.deny("dev", ["models.manage"])
    assert not apps.authenticate(token).has("models.manage")
    apps.revoke("dev")
    assert apps.authenticate(token) is None
    with pytest.raises(AuthError) as e:
        apps.grant("dev", ["chat"])
    assert e.value.code == "app_not_found"


def test_reregistration_and_invalid_input(reg):
    _, _, apps = reg
    apps.register("dev")
    with pytest.raises(AuthError) as e:
        apps.register("dev")
    assert e.value.code == "app_exists"
    for bad in ("", "A", "UPPER", "has space", "../x", "x" * 60):
        with pytest.raises(AuthError):
            apps.register(bad)
    with pytest.raises(AuthError) as e:
        apps.grant("dev", ["make.coffee"])
    assert e.value.code == "unknown_permission"


def test_principal_permission_error_tells_how_to_fix(reg):
    _, _, apps = reg
    _, token = apps.register("dev")
    with pytest.raises(AuthError) as e:
        apps.authenticate(token).require("models.manage")
    assert e.value.status == 403 and "anvira app grant dev models.manage" in e.value.message


@pytest.mark.parametrize("text", [
    "Authorization: Bearer abcdefghijklmnop1234", "api_key=sk-abcdefghijklmnopqrstuv", 'password: "hunter2hunter2"',
    "X-Orcha-Token: 0123456789abcdef0123456789abcdef", "token=hf_abcdefghijklmnopqrstu"])
def test_redaction_removes_secrets(text):
    out = redact(text)
    assert "[REDACTED]" in out
    for secret in ("abcdefghijklmnop1234", "sk-abcdefghijklmnopqrstuv", "hunter2hunter2", "0123456789abcdef0123456789abcdef", "hf_abcdefghijklmnopqrstu"):
        assert secret not in out


def test_redact_obj_masks_by_key():
    o = redact_obj({"api_key": "s3cretvalue", "nested": [{"Authorization": "Bearer xyz12345678"}], "ok": "fine"})
    assert o["api_key"] == "[REDACTED]" and o["nested"][0]["Authorization"] == "[REDACTED]" and o["ok"] == "fine"
