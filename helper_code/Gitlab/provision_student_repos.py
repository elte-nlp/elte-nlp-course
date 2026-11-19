#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["python-gitlab>=4.0", "pydantic>=2.0"]
# ///
"""Create one private GitLab repository per student inside a group.

Each student is added to *their own* repository only, never to the group, so
students cannot see each other's work. Members of the group (teachers, TAs)
keep access to every repository through inheritance.

What the script does for every row of the students CSV:
    1. Creates ``<group>/<prefix><username>`` as a private project with a README
       (skipped if the project already exists).
    2. Lets Developers push to the default branch, which GitLab protects for
       Maintainers only by default (skip with ``--keep-branch-protection``).
    3. Adds the student as a project member (``username`` column) or sends a
       project invitation by e-mail (``email`` column, used when there is no
       username; skip with ``--no-invite``). Existing members and pending
       invitations are left alone.

If a username has no GitLab account, the repository is still created and the
student is reported as failed. With ``--delete-if-no-user`` the repository is
deleted instead (or not created), unless it holds commits beyond the initial
README.

The script is idempotent: re-running it only does what is still missing, so it
is safe to run again after adding students to the CSV or after a failure.

Output is coloured on terminals: green ✓ for changes made, red ✗ for failures,
yellow ! for warnings. ``--no-color`` or the ``NO_COLOR`` variable turns it off.

Students CSV (UTF-8, header row required; ``username`` or ``email`` per row)::

    username,email,name,repo
    kovacs.adam,,Kovács Ádám,
    ,nagy.eva@example.edu,Nagy Éva,
    szabo_p,,Szabó Péter,team-szabo

    username  GitLab username; the student is added directly.
    email     Used only when username is empty; GitLab sends an invitation
              unless ``--no-invite`` is given.
    name      Optional, used in the project description.
    repo      Optional full repository path; overrides ``--prefix`` + username.

Usage::

    export GITLAB_TOKEN=glpat-...      # `api` scope, Owner or Maintainer of the group
    uv run provision_student_repos.py \\
        --url https://gitlab.example.edu \\
        --group courses/prog1-2026 \\
        --students students.csv \\
        --prefix hw1- \\
        --dry-run                       # drop --dry-run to apply

Requires ``python-gitlab`` and ``pydantic``; ``uv run`` installs them automatically.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

import gitlab
from gitlab.const import AccessLevel
from gitlab.exceptions import GitlabError, GitlabGetError, GitlabInvitationError, GitlabListError
from gitlab.v4.objects import Group, Project
from pydantic import BaseModel, SecretStr, ValidationError, field_validator, model_validator

logger = logging.getLogger("provision_student_repos")

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

ACCESS_LEVELS: dict[str, AccessLevel] = {
    "reporter": AccessLevel.REPORTER,
    "developer": AccessLevel.DEVELOPER,
    "maintainer": AccessLevel.MAINTAINER,
}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_INVALID_PATH_CHARS = re.compile(r"[^a-z0-9_.-]+")
_REPEATED_DASHES = re.compile(r"-{2,}")

_ANSI_RESET = "\033[0m"
# log level -> (mark, ASCII fallback mark, ANSI colour)
_LEVEL_STYLES: dict[int, tuple[str, str, str]] = {
    logging.DEBUG: ("·", ".", "\033[2m"),  # dim
    logging.INFO: ("•", "-", ""),
    SUCCESS: ("✓", "+", "\033[32m"),  # green
    logging.WARNING: ("!", "!", "\033[33m"),  # yellow
    logging.ERROR: ("✗", "x", "\033[31m"),  # red
}


class ProvisioningError(Exception):
    """Raised when the input is invalid or a student's repository cannot be set up."""


class ConsoleFormatter(logging.Formatter):
    """Prefix each log line with a status mark and colour it by level."""

    def __init__(self, *, color: bool, unicode_marks: bool) -> None:
        super().__init__("%(message)s")
        self._color = color
        self._unicode_marks = unicode_marks

    def format(self, record: logging.LogRecord) -> str:
        """Return the message with its mark, coloured if enabled."""
        level = max((lvl for lvl in _LEVEL_STYLES if lvl <= record.levelno), default=logging.DEBUG)
        mark, ascii_mark, ansi = _LEVEL_STYLES[level]
        line = f"{mark if self._unicode_marks else ascii_mark} {super().format(record)}"
        if self._color and ansi:
            return f"{ansi}{line}{_ANSI_RESET}"
        return line


class Settings(BaseModel):
    """Run configuration, built from the command line and the environment."""

    url: str
    token: SecretStr
    group: str
    students_csv: Path
    prefix: str = ""
    access_level: AccessLevel = AccessLevel.DEVELOPER
    expires_at: date | None = None
    open_default_branch: bool = True
    send_invitations: bool = True
    delete_if_no_user: bool = False
    dry_run: bool = False


class Student(BaseModel):
    """One row of the students CSV."""

    username: str | None = None
    email: str | None = None
    name: str | None = None
    repo: str | None = None

    @field_validator("username", "email", "name", "repo", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("username")
    @classmethod
    def _strip_at_sign(cls, value: str | None) -> str | None:
        return value.lstrip("@") if value else value

    @field_validator("email")
    @classmethod
    def _check_email(cls, value: str | None) -> str | None:
        if value is not None and not _EMAIL_RE.match(value):
            raise ValueError(f"not a valid e-mail address: {value!r}")
        return value

    @model_validator(mode="after")
    def _require_identity(self) -> Student:
        if self.username is None and self.email is None:
            raise ValueError("needs a username or an email")
        return self

    @property
    def label(self) -> str:
        """Short identifier for log messages."""
        return self.username or self.email or "?"

    def repo_path(self, prefix: str) -> str:
        """Return the repository path (URL slug) for this student."""
        if self.repo:
            return slugify(self.repo)
        base = self.username or (self.email or "").split("@", 1)[0]
        return slugify(f"{prefix}{base}")


@dataclass(frozen=True)
class Assignment:
    """A student paired with the repository path they will get."""

    student: Student
    path: str


def slugify(text: str) -> str:
    """Turn text into a valid GitLab project path (``Kovács Ádám`` -> ``kovacs-adam``)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _INVALID_PATH_CHARS.sub("-", ascii_text.lower())
    slug = _REPEATED_DASHES.sub("-", slug).strip("-._")
    if not slug:
        raise ProvisioningError(f"cannot build a repository path from {text!r}")
    return slug


def load_students(csv_path: Path) -> list[Student]:
    """Read and validate the students CSV; report every bad row at once."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = {(name or "").strip().lower() for name in reader.fieldnames or []}
        if not columns & {"username", "email"}:
            raise ProvisioningError(f"{csv_path}: header must contain a 'username' or 'email' column")

        students: list[Student] = []
        errors: list[str] = []
        for row in reader:
            normalized = {key.strip().lower(): value for key, value in row.items() if key is not None}
            try:
                students.append(Student.model_validate(normalized))
            except ValidationError as exc:
                errors.append(f"{csv_path}, line {reader.line_num}: {exc.errors()[0]['msg']}")

    if errors:
        raise ProvisioningError("invalid rows in the students CSV:\n  " + "\n  ".join(errors))
    if not students:
        raise ProvisioningError(f"{csv_path}: no students found")
    return students


def build_assignments(students: list[Student], prefix: str) -> list[Assignment]:
    """Pair each student with a repository path; refuse collisions."""
    owners: dict[str, Student] = {}
    assignments: list[Assignment] = []
    for student in students:
        path = student.repo_path(prefix)
        if path in owners:
            raise ProvisioningError(
                f"{student.label} and {owners[path].label} would both get repository "
                f"{path!r}; set the 'repo' column for one of them"
            )
        owners[path] = student
        assignments.append(Assignment(student=student, path=path))
    return assignments


def _group_not_found_message(gl: gitlab.Gitlab, group_arg: str) -> str:
    message = (
        f"group {group_arg!r} not found. --group needs the group's path as it appears in its URL "
        "(e.g. my-course/2026-27-1) or its numeric group ID, not its display name."
    )
    search_term = group_arg.rsplit("/", 1)[-1].strip()
    if not search_term:
        return message
    try:
        candidates = gl.groups.list(search=search_term, per_page=20, get_all=False)
    except GitlabError:
        return message
    if not candidates:
        return message
    lines = [f"  --group {group.full_path}   (ID {group.id}, name: {group.full_name})" for group in candidates]
    return f"{message}\nGroups matching {search_term!r} that you can see:\n" + "\n".join(lines)


def _is_scheduled_for_deletion(project: Project) -> bool:
    attributes = project.attributes
    return bool(attributes.get("marked_for_deletion_at") or attributes.get("marked_for_deletion_on"))


def _has_commits_beyond_initial(project: Project) -> bool:
    try:
        commits = project.commits.list(per_page=2, get_all=False, query_parameters={"all": "true"})
    except GitlabListError as exc:
        if exc.response_code == 404:  # empty repository
            return False
        raise
    return len(commits) > 1


class Provisioner:
    """Creates student repositories in one group and grants access to them."""

    def __init__(self, gl: gitlab.Gitlab, group: Group, settings: Settings) -> None:
        self._gl = gl
        self._group = group
        self._settings = settings

    @classmethod
    def connect(cls, settings: Settings) -> Provisioner:
        """Sign in to GitLab and look up the target group."""
        gl = gitlab.Gitlab(
            settings.url,
            private_token=settings.token.get_secret_value(),
            retry_transient_errors=True,
        )
        gl.auth()
        try:
            group = gl.groups.get(settings.group)
        except GitlabGetError as exc:
            if exc.response_code != 404:
                raise
            raise ProvisioningError(_group_not_found_message(gl, settings.group)) from exc
        user = gl.user.username if gl.user else "?"
        logger.info("Signed in as %s, target group: %s", user, group.full_path)
        return cls(gl, group, settings)

    def warn_about_group_members(self, assignments: list[Assignment]) -> None:
        """Warn if a student is a group member, since they would see every repository."""
        try:
            members = self._group.members_all.list(get_all=True)
        except GitlabError as exc:
            logger.warning("Could not check group membership: %s", exc)
            return
        group_usernames = {member.username for member in members}
        for assignment in assignments:
            username = assignment.student.username
            if username in group_usernames:
                logger.warning(
                    "%s is a member of group %s (or a parent group) and can see ALL student "
                    "repositories; remove them from the group",
                    username,
                    self._group.full_path,
                )

    def provision_all(self, assignments: list[Assignment]) -> list[str]:
        """Provision every assignment; return the labels of the students that failed."""
        failed: list[str] = []
        for assignment in assignments:
            try:
                self.provision(assignment)
            except (GitlabError, ProvisioningError) as exc:
                logger.error("%s: FAILED: %s", assignment.path, exc)
                failed.append(assignment.student.label)
        return failed

    def provision(self, assignment: Assignment) -> None:
        """Create the student's repository if needed and give the student access."""
        username = assignment.student.username
        user_id = self._find_user_id(username) if username else None
        missing_user = username is not None and user_id is None

        if missing_user and self._settings.delete_if_no_user:
            outcome = self._delete_project(assignment)
            raise ProvisioningError(f"no GitLab user with username {username!r}; {outcome}")

        project = self._ensure_project(assignment)
        if project is not None and self._settings.open_default_branch:
            self._open_default_branch(project, assignment.path)
        if missing_user:
            raise ProvisioningError(
                f"no GitLab user with username {username!r}; the repository is kept "
                "(fix the username and re-run, or use --delete-if-no-user to remove it)"
            )
        if project is None:
            logger.info("%s: [dry run] would %s", assignment.path, self._planned_access(assignment.student))
            return
        self._grant_access(project, assignment, user_id)

    def _grant_access(self, project: Project, assignment: Assignment, user_id: int | None) -> None:
        student = assignment.student
        if student.username and user_id is not None:
            self._add_member(project, assignment.path, student.username, user_id)
        elif self._settings.send_invitations:
            self._invite(project, assignment.path, student.email or "")
        else:
            logger.info("%s: no invitation sent to %s (--no-invite)", assignment.path, student.email)

    def _planned_access(self, student: Student) -> str:
        if student.username:
            return f"add {student.username}"
        if self._settings.send_invitations:
            return f"invite {student.email}"
        return f"not invite {student.email} (--no-invite)"

    def _full_path(self, assignment: Assignment) -> str:
        return f"{self._group.full_path}/{assignment.path}"

    def _get_project(self, full_path: str) -> Project | None:
        try:
            return self._gl.projects.get(full_path)
        except GitlabGetError as exc:
            if exc.response_code != 404:
                raise
            return None

    def _find_user_id(self, username: str) -> int | None:
        users = self._gl.users.list(username=username, get_all=False)
        return int(users[0].id) if users else None

    def _ensure_project(self, assignment: Assignment) -> Project | None:
        full_path = self._full_path(assignment)
        project = self._get_project(full_path)
        if project is not None:
            if _is_scheduled_for_deletion(project):
                raise ProvisioningError(
                    f"{full_path} is scheduled for deletion; restore it in GitLab or wait until it is removed"
                )
            logger.info("%s: project already exists", assignment.path)
            if project.visibility != "private":
                logger.warning("%s: existing project is %s, not private", assignment.path, project.visibility)
            return project

        if self._settings.dry_run:
            logger.info("%s: [dry run] would create private project %s", assignment.path, full_path)
            return None

        student = assignment.student
        project = self._gl.projects.create(
            {
                "name": assignment.path,
                "path": assignment.path,
                "namespace_id": self._group.id,
                "visibility": "private",
                "initialize_with_readme": True,
                "description": f"Repository of {student.name or student.label}",
            }
        )
        logger.log(SUCCESS, "%s: created project %s", assignment.path, project.web_url)
        return project

    def _delete_project(self, assignment: Assignment) -> str:
        """Delete the student's repository if it is safe to; return what happened."""
        full_path = self._full_path(assignment)
        project = self._get_project(full_path)
        if project is None:
            return "no repository was created"
        if _is_scheduled_for_deletion(project):
            return f"repository {full_path} is already scheduled for deletion"
        if _has_commits_beyond_initial(project):
            return (
                f"repository {full_path} was NOT deleted because it contains commits beyond the initial "
                "README (delete it by hand if you are sure)"
            )
        if self._settings.dry_run:
            return f"[dry run] would delete repository {full_path}"
        project.delete()
        return f"deleted repository {full_path}"

    def _open_default_branch(self, project: Project, path: str) -> None:
        if self._settings.access_level >= AccessLevel.MAINTAINER:
            return
        branch = project.default_branch
        if not branch:
            logger.warning("%s: repository has no default branch yet; branch protection left as is", path)
            return
        try:
            protected = project.protectedbranches.get(branch)
        except GitlabGetError as exc:
            if exc.response_code != 404:
                raise
            logger.debug("%s: branch %s is not protected", path, branch)
            return

        push_levels = [rule.get("access_level") or 0 for rule in protected.push_access_levels]
        if any(0 < level <= AccessLevel.DEVELOPER for level in push_levels):
            logger.debug("%s: Developers can already push to %s", path, branch)
            return
        if self._settings.dry_run:
            logger.info("%s: [dry run] would allow Developers to push to %s", path, branch)
            return

        project.protectedbranches.delete(branch)
        project.protectedbranches.create(
            {
                "name": branch,
                "push_access_level": int(AccessLevel.DEVELOPER),
                "merge_access_level": int(AccessLevel.DEVELOPER),
                "allow_force_push": False,
            }
        )
        logger.log(SUCCESS, "%s: Developers may now push to %s (force push still blocked)", path, branch)

    def _add_member(self, project: Project, path: str, username: str, user_id: int) -> None:
        try:
            member = project.members.get(user_id)
        except GitlabGetError as exc:
            if exc.response_code != 404:
                raise
        else:
            logger.info("%s: %s is already a member (access level %s)", path, username, member.access_level)
            return

        if self._settings.dry_run:
            logger.info("%s: [dry run] would add %s", path, username)
            return
        project.members.create(self._membership_data({"user_id": user_id}))
        logger.log(SUCCESS, "%s: added %s as %s", path, username, self._settings.access_level.name.title())

    def _invite(self, project: Project, path: str, email: str) -> None:
        pending = project.invitations.list(query=email, get_all=True)
        if any(str(invite.attributes.get("invite_email", "")).lower() == email.lower() for invite in pending):
            logger.info("%s: %s already has a pending invitation", path, email)
            return

        if self._settings.dry_run:
            logger.info("%s: [dry run] would invite %s", path, email)
            return
        try:
            project.invitations.create(self._membership_data({"email": email}))
        except GitlabInvitationError as exc:
            if "already" in str(exc).lower():
                logger.info("%s: %s is already a member or invited (%s)", path, email, exc)
                return
            raise
        logger.log(SUCCESS, "%s: invited %s as %s", path, email, self._settings.access_level.name.title())

    def _membership_data(self, who: dict[str, str | int]) -> dict[str, str | int]:
        data: dict[str, str | int] = {**who, "access_level": int(self._settings.access_level)}
        if self._settings.expires_at:
            data["expires_at"] = self._settings.expires_at.isoformat()
        return data


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a private GitLab repository per student in a group and give "
        "each student access to their own repository only.",
        epilog="The access token is read from the GITLAB_TOKEN environment variable.",
    )
    parser.add_argument("--url", default=os.environ.get("GITLAB_URL"), help="GitLab base URL (or GITLAB_URL)")
    parser.add_argument("--group", required=True, help="group path (e.g. courses/prog1-2026) or numeric ID")
    parser.add_argument("--students", required=True, type=Path, help="students CSV file")
    parser.add_argument("--prefix", default="", help="repository name prefix, e.g. 'hw1-'")
    parser.add_argument(
        "--access-level",
        choices=sorted(ACCESS_LEVELS),
        default="developer",
        help="role students get in their repository (default: developer)",
    )
    parser.add_argument(
        "--expires-at",
        type=date.fromisoformat,
        help="YYYY-MM-DD date when student access ends (e.g. end of semester)",
    )
    parser.add_argument(
        "--keep-branch-protection",
        action="store_true",
        help="do not allow Developers to push to the default branch",
    )
    parser.add_argument(
        "--no-invite",
        action="store_true",
        help="create repositories for email-only students without sending them an invitation "
        "(re-run without this flag to invite them later)",
    )
    parser.add_argument(
        "--delete-if-no-user",
        action="store_true",
        help="if a username has no GitLab account, delete that student's repository (or don't create it); "
        "repositories with commits beyond the initial README are never deleted",
    )
    parser.add_argument("--dry-run", action="store_true", help="show what would happen without changing anything")
    parser.add_argument("--no-color", action="store_true", help="plain output without colours (or set NO_COLOR)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def _can_encode(stream: TextIO, text: str) -> bool:
    try:
        text.encode(stream.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _configure_logging(*, verbose: bool, color: bool) -> None:
    stream = sys.stderr
    use_color = color and not os.environ.get("NO_COLOR") and stream.isatty()
    unicode_marks = _can_encode(stream, "".join(mark for mark, _, _ in _LEVEL_STYLES.values()))
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ConsoleFormatter(color=use_color, unicode_marks=unicode_marks))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler])
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _prepare(args: argparse.Namespace) -> tuple[Provisioner, list[Assignment]]:
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise ProvisioningError("set the GITLAB_TOKEN environment variable to a token with the 'api' scope")
    if not args.url:
        raise ProvisioningError("pass --url or set GITLAB_URL")

    settings = Settings(
        url=args.url,
        token=SecretStr(token),
        group=args.group,
        students_csv=args.students,
        prefix=args.prefix,
        access_level=ACCESS_LEVELS[args.access_level],
        expires_at=args.expires_at,
        open_default_branch=not args.keep_branch_protection,
        send_invitations=not args.no_invite,
        delete_if_no_user=args.delete_if_no_user,
        dry_run=args.dry_run,
    )
    assignments = build_assignments(load_students(settings.students_csv), settings.prefix)
    return Provisioner.connect(settings), assignments


def main(argv: list[str] | None = None) -> int:
    """Run the script; return 0 on success, 1 if some students failed, 2 on setup errors."""
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose, color=not args.no_color)

    try:
        provisioner, assignments = _prepare(args)
    except (ProvisioningError, GitlabError, OSError) as exc:
        logger.error("%s", exc)
        return 2

    if args.dry_run:
        logger.warning("DRY RUN: nothing will be changed")
    provisioner.warn_about_group_members(assignments)
    failed = provisioner.provision_all(assignments)

    total = len(assignments)
    not_invited = sum(1 for assignment in assignments if not assignment.student.username)
    if args.no_invite and not_invited:
        logger.info("%d email-only students were not invited; re-run without --no-invite to invite them", not_invited)
    if failed:
        logger.error("Finished: %d of %d students OK; failed: %s", total - len(failed), total, ", ".join(failed))
        logger.error("Fix the problems above and re-run; students that are already set up are skipped")
        return 1
    logger.log(SUCCESS, "Finished: all %d students OK", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
