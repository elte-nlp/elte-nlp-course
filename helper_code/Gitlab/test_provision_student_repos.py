import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import responses
from responses import matchers

from provision_student_repos import (
    SUCCESS,
    ConsoleFormatter,
    ProvisioningError,
    build_assignments,
    load_students,
    main,
    slugify,
)

API = "https://gitlab.test/api/v4"
GROUP = {"id": 7, "full_path": "courses/prog1", "name": "prog1"}


def _write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "students.csv"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-test")


@pytest.fixture
def api() -> Iterator[responses.RequestsMock]:
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.get(f"{API}/user", json={"id": 1, "username": "teacher"})
        rsps.get(f"{API}/groups/courses%2Fprog1", json=GROUP)
        rsps.get(f"{API}/groups/7/members/all", json=[{"id": 1, "username": "teacher"}])
        yield rsps


def _run(csv_path: Path, *extra: str) -> int:
    return main(["--url", "https://gitlab.test", "--group", "courses/prog1", "--students", str(csv_path), *extra])


def _bodies(api: responses.RequestsMock, method: str) -> list[tuple[str, dict[str, object]]]:
    return [
        (str(call.request.url), json.loads(cast(str | bytes | None, call.request.body) or "{}"))
        for call in api.calls
        if call.request.method == method
    ]


def test_slugify_strips_accents_and_invalid_characters() -> None:
    assert slugify("hw1-Kovács Ádám") == "hw1-kovacs-adam"
    assert slugify("--szabo_p.") == "szabo_p"


def test_load_students_reports_every_bad_row(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "username,email\nalice,\n,\n,not-an-email\n")
    with pytest.raises(ProvisioningError) as excinfo:
        load_students(csv_path)
    assert "line 3" in str(excinfo.value)
    assert "line 4" in str(excinfo.value)


def test_load_students_accepts_excel_bom_and_at_sign(tmp_path: Path) -> None:
    csv_path = tmp_path / "students.csv"
    csv_path.write_text("﻿Username,Email\n@alice,\n", encoding="utf-8")
    assert load_students(csv_path)[0].username == "alice"


def test_build_assignments_refuses_path_collisions(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "username,email\n,john@a.edu\n,john@b.edu\n")
    with pytest.raises(ProvisioningError, match="would both get repository 'hw1-john'"):
        build_assignments(load_students(csv_path), "hw1-")


def test_creates_project_opens_branch_and_adds_member(tmp_path: Path, token: None, api: responses.RequestsMock) -> None:
    csv_path = _write_csv(tmp_path, "username,name\nalice,Alice A\n")
    api.get(f"{API}/projects/courses%2Fprog1%2Fhw1-alice", status=404, json={"message": "404 Project Not Found"})
    api.post(
        f"{API}/projects",
        json={"id": 42, "default_branch": "main", "web_url": "https://gitlab.test/courses/prog1/hw1-alice"},
    )
    api.get(
        f"{API}/projects/42/protected_branches/main",
        json={"name": "main", "push_access_levels": [{"access_level": 40}]},
    )
    api.delete(f"{API}/projects/42/protected_branches/main", status=204)
    api.post(f"{API}/projects/42/protected_branches", json={"name": "main"})
    api.get(f"{API}/users", match=[matchers.query_param_matcher({"username": "alice"})], json=[{"id": 99}])
    api.get(f"{API}/projects/42/members/99", status=404, json={"message": "404 Not found"})
    api.post(f"{API}/projects/42/members", json={"id": 99, "access_level": 30})

    assert _run(csv_path, "--prefix", "hw1-", "--expires-at", "2027-02-01") == 0

    posts = _bodies(api, "POST")
    assert posts[0] == (
        f"{API}/projects",
        {
            "name": "hw1-alice",
            "path": "hw1-alice",
            "namespace_id": 7,
            "visibility": "private",
            "initialize_with_readme": True,
            "description": "Repository of Alice A",
        },
    )
    assert posts[1][1] == {"name": "main", "push_access_level": 30, "merge_access_level": 30, "allow_force_push": False}
    assert posts[2][1] == {"user_id": 99, "access_level": 30, "expires_at": "2027-02-01"}


def test_rerun_changes_nothing_when_everything_exists(tmp_path: Path, token: None, api: responses.RequestsMock) -> None:
    csv_path = _write_csv(tmp_path, "username\nalice\n")
    api.get(
        f"{API}/projects/courses%2Fprog1%2Falice",
        json={"id": 42, "default_branch": "main", "visibility": "private"},
    )
    api.get(
        f"{API}/projects/42/protected_branches/main",
        json={"name": "main", "push_access_levels": [{"access_level": 30}]},
    )
    api.get(f"{API}/users", json=[{"id": 99}])
    api.get(f"{API}/projects/42/members/99", json={"id": 99, "access_level": 30})

    assert _run(csv_path) == 0
    assert _bodies(api, "POST") == []
    assert _bodies(api, "DELETE") == []


def test_email_only_student_gets_an_invitation(tmp_path: Path, token: None, api: responses.RequestsMock) -> None:
    csv_path = _write_csv(tmp_path, "username,email\n,Eva.Nagy@uni.edu\n")
    api.get(
        f"{API}/projects/courses%2Fprog1%2Feva.nagy",
        json={"id": 43, "default_branch": "main", "visibility": "private"},
    )
    api.get(f"{API}/projects/43/protected_branches/main", status=404, json={"message": "404 Not Found"})
    api.get(f"{API}/projects/43/invitations", json=[])
    api.post(f"{API}/projects/43/invitations", json={"status": "success"})

    assert _run(csv_path) == 0
    assert _bodies(api, "POST") == [
        (f"{API}/projects/43/invitations", {"email": "Eva.Nagy@uni.edu", "access_level": 30})
    ]


def test_failed_invitation_is_reported_and_others_continue(
    tmp_path: Path, token: None, api: responses.RequestsMock
) -> None:
    csv_path = _write_csv(tmp_path, "username,email\n,bad@uni.edu\nbob,\n")
    api.get(
        f"{API}/projects/courses%2Fprog1%2Fbad",
        json={"id": 50, "default_branch": "main", "visibility": "private"},
    )
    api.get(f"{API}/projects/50/protected_branches/main", status=404, json={"message": "404 Not Found"})
    api.get(f"{API}/projects/50/invitations", json=[])
    api.post(
        f"{API}/projects/50/invitations",
        status=201,
        json={"status": "error", "message": {"bad@uni.edu": "Invite is blocked by domain restrictions"}},
    )
    api.get(
        f"{API}/projects/courses%2Fprog1%2Fbob",
        json={"id": 51, "default_branch": "main", "visibility": "private"},
    )
    api.get(f"{API}/projects/51/protected_branches/main", status=404, json={"message": "404 Not Found"})
    api.get(f"{API}/users", json=[{"id": 5}])
    api.get(f"{API}/projects/51/members/5", status=404, json={"message": "404 Not found"})
    api.post(f"{API}/projects/51/members", json={"id": 5})

    assert _run(csv_path) == 1


def test_dry_run_makes_no_changes(tmp_path: Path, token: None, api: responses.RequestsMock) -> None:
    csv_path = _write_csv(tmp_path, "username\nalice\n")
    api.get(f"{API}/users", json=[{"id": 99}])
    api.get(f"{API}/projects/courses%2Fprog1%2Falice", status=404, json={"message": "404 Project Not Found"})

    assert _run(csv_path, "--dry-run") == 0
    assert all(call.request.method == "GET" for call in api.calls)


def test_warns_when_student_is_a_group_member(tmp_path: Path, token: None, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = _write_csv(tmp_path, "username\nalice\n")
    with responses.RequestsMock() as api:
        api.get(f"{API}/user", json={"id": 1, "username": "teacher"})
        api.get(f"{API}/groups/courses%2Fprog1", json=GROUP)
        api.get(f"{API}/groups/7/members/all", json=[{"id": 99, "username": "alice"}])
        api.get(f"{API}/users", json=[{"id": 99}])
        api.get(f"{API}/projects/courses%2Fprog1%2Falice", status=404, json={"message": "404"})
        assert _run(csv_path, "--dry-run") == 0
    assert "can see ALL student repositories" in caplog.text


def test_missing_token_is_a_setup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    assert _run(_write_csv(tmp_path, "username\nalice\n")) == 2


def test_no_invite_creates_repo_but_sends_no_invitation(
    tmp_path: Path, token: None, api: responses.RequestsMock
) -> None:
    csv_path = _write_csv(tmp_path, "username,email\n,eva@uni.edu\n")
    api.get(f"{API}/projects/courses%2Fprog1%2Feva", status=404, json={"message": "404 Project Not Found"})
    api.post(f"{API}/projects", json={"id": 44, "default_branch": "main", "web_url": "https://gitlab.test/x"})
    api.get(f"{API}/projects/44/protected_branches/main", status=404, json={"message": "404 Not Found"})

    assert _run(csv_path, "--no-invite") == 0
    assert [url for url, _ in _bodies(api, "POST")] == [f"{API}/projects"]
    assert not any("invitations" in str(call.request.url) for call in api.calls)


def test_group_display_name_gets_a_helpful_error(tmp_path: Path, token: None, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = _write_csv(tmp_path, "username\nalice\n")
    display_name = "Natural Language Processing and Foundation Models / 2026-27-1"
    with responses.RequestsMock() as api:
        api.get(f"{API}/user", json={"id": 1, "username": "teacher"})
        api.get(re.compile(rf"{API}/groups/Natural.*"), status=404, json={"message": "404 Group Not Found"})
        api.get(
            f"{API}/groups",
            match=[matchers.query_param_matcher({"search": "2026-27-1", "per_page": "20"})],
            json=[{"id": 321, "full_path": "nlp-fm/2026-27-1", "full_name": display_name}],
        )
        exit_code = main(["--url", "https://gitlab.test", "--group", display_name, "--students", str(csv_path)])
    assert exit_code == 2
    assert "--group nlp-fm/2026-27-1   (ID 321" in caplog.text


def _missing_user_mocks(api: responses.RequestsMock, commits: int) -> None:
    api.get(f"{API}/users", match=[matchers.query_param_matcher({"username": "ghost"})], json=[])
    api.get(
        f"{API}/projects/courses%2Fprog1%2Fghost",
        json={"id": 60, "default_branch": "main", "visibility": "private"},
    )
    api.get(
        f"{API}/projects/60/repository/commits",
        match=[matchers.query_param_matcher({"per_page": "2", "all": "true"})],
        json=[{"id": f"sha{i}"} for i in range(commits)],
    )


def test_missing_user_keeps_repository_by_default(
    tmp_path: Path, token: None, api: responses.RequestsMock, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = _write_csv(tmp_path, "username\nghost\n")
    api.get(f"{API}/users", json=[])
    api.get(f"{API}/projects/courses%2Fprog1%2Fghost", status=404, json={"message": "404 Project Not Found"})
    api.post(f"{API}/projects", json={"id": 60, "default_branch": "main", "web_url": "https://gitlab.test/x"})
    api.get(f"{API}/projects/60/protected_branches/main", status=404, json={"message": "404 Not Found"})

    assert _run(csv_path) == 1
    assert [url for url, _ in _bodies(api, "POST")] == [f"{API}/projects"]
    assert "the repository is kept" in caplog.text


def test_delete_if_no_user_deletes_untouched_repository(
    tmp_path: Path, token: None, api: responses.RequestsMock, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = _write_csv(tmp_path, "username\nghost\n")
    _missing_user_mocks(api, commits=1)
    api.delete(f"{API}/projects/60", status=202, json={"message": "202 Accepted"})

    assert _run(csv_path, "--delete-if-no-user") == 1
    assert [url for url, _ in _bodies(api, "DELETE")] == [f"{API}/projects/60"]
    assert "deleted repository courses/prog1/ghost" in caplog.text


def test_delete_if_no_user_never_deletes_repository_with_work(
    tmp_path: Path, token: None, api: responses.RequestsMock, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = _write_csv(tmp_path, "username\nghost\n")
    _missing_user_mocks(api, commits=2)

    assert _run(csv_path, "--delete-if-no-user") == 1
    assert _bodies(api, "DELETE") == []
    assert "was NOT deleted" in caplog.text


def test_delete_if_no_user_does_not_create_repository(tmp_path: Path, token: None, api: responses.RequestsMock) -> None:
    csv_path = _write_csv(tmp_path, "username\nghost\n")
    api.get(f"{API}/users", json=[])
    api.get(f"{API}/projects/courses%2Fprog1%2Fghost", status=404, json={"message": "404 Project Not Found"})

    assert _run(csv_path, "--delete-if-no-user") == 1
    assert all(call.request.method == "GET" for call in api.calls)


def test_delete_if_no_user_dry_run_deletes_nothing(
    tmp_path: Path, token: None, api: responses.RequestsMock, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = _write_csv(tmp_path, "username\nghost\n")
    _missing_user_mocks(api, commits=1)

    assert _run(csv_path, "--delete-if-no-user", "--dry-run") == 1
    assert all(call.request.method == "GET" for call in api.calls)
    assert "[dry run] would delete repository courses/prog1/ghost" in caplog.text


def _record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord("test", level, __file__, 1, message, None, None)


def test_console_formatter_colours_and_marks() -> None:
    formatter = ConsoleFormatter(color=True, unicode_marks=True)
    assert formatter.format(_record(SUCCESS, "created")) == "\033[32m✓ created\033[0m"
    assert formatter.format(_record(logging.ERROR, "failed")) == "\033[31m✗ failed\033[0m"
    assert formatter.format(_record(logging.INFO, "exists")) == "• exists"


def test_console_formatter_plain_ascii() -> None:
    formatter = ConsoleFormatter(color=False, unicode_marks=False)
    assert formatter.format(_record(SUCCESS, "created")) == "+ created"
    assert formatter.format(_record(logging.CRITICAL, "boom")) == "x boom"
