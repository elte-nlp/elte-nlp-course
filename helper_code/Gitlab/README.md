# Student repository provisioning for GitLab

`provision_student_repos.py` creates one private GitLab repository per student inside a course group, and gives each student access to **their own repository only**.

Students are never added to the group itself. Group members see every project in the group, so adding students there would let them see each other's work. Instead, each student becomes a member of just their own repository. You, as group Owner, and anyone else you add to the group (e.g. TAs) can see all of the repositories through the group.

## What it does

For every student in the CSV, the script:

1. **Creates the repository** `<group>/<prefix><username>`: private, with a README, so the default branch exists from the start.
2. **Lets students push to the default branch.** By default GitLab lets only Maintainers push to it. The script allows Developers to push and merge there, but force pushes stay blocked. Turn this off with `--keep-branch-protection`.
3. **Gives the student access.**
   - If the CSV row has a **username**, the student is added to the repository directly.
   - If the row has only an **email**, GitLab sends a project invitation. If that email belongs to an existing GitLab account, the account is added directly. Turn invitations off with `--no-invite`.

The script is **safe to re-run**. Repositories, members and pending invitations that already exist are left alone, so you can add students to the CSV later or re-run after a failure. If one student fails, the others are still processed, and the failures are listed at the end.

**When a username has no GitLab account** (a typo, or the student hasn't registered yet), the student is reported as failed. By default the repository is still created and kept, so fixing the username or waiting for the student to register and then re-running finishes the job. With `--delete-if-no-user`, such repositories are deleted instead, or not created at all. As a safety measure, the script never deletes a repository that contains any commit besides the initial README. That protects, for example, the work of a student who renamed their GitLab account mid-semester.

It also **warns you if a listed student is a member of the group** (or a parent group), because that student can see every repository.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- The Python packages `python-gitlab` and `pydantic`, which are listed in `requirements.txt` and inside the script itself
- A GitLab personal access token with the `api` scope, created by a user who is Owner or Maintainer of the group. Deleting repositories (`--delete-if-no-user`) needs the Owner role.

## Setup

### 1. Create a GitLab access token

1. In GitLab, select your avatar in the upper-right corner, then **Edit profile**.
2. In the left sidebar, select **Access** > **Personal access tokens**. You can also open `https://<your-gitlab>/-/user_settings/personal_access_tokens` directly.
3. If there is a **Generate token** dropdown, choose **Legacy token**.
4. Enter a name, set a short expiration date (a week or two is plenty), and tick the **`api`** scope. `read_api` is not enough.
5. Select **Generate token** and copy it right away. GitLab only shows it once.

Treat the token like a password: it can do anything you can do in GitLab. Don't save it in the script or commit it to git, and revoke it when you're done.

### 2. Find the group's path or ID

`--group` takes the group's **path** or **numeric ID**, not its display name.

- **Path:** open the group in your browser. Everything after the hostname in the address bar is the path. For `https://gitlab.example.edu/nlp-fm/2026-27-1` it is `nlp-fm/2026-27-1`.
- **ID:** shown on the group's overview page (as "Group ID", or under **Copy group ID** in the ⋮ menu, depending on your GitLab version).

If you pass a display name by mistake, the script lists the matching groups with the correct `--group` value.

### 3. Prepare the students CSV

The file must be UTF-8 (files saved from Excel as "CSV UTF-8" work) and must have a header row. Every row needs a `username` or an `email`.

```csv
username,email,name,repo
kovacs.adam,,Kovács Ádám,
,nagy.eva@example.edu,Nagy Éva,
szabo_p,,Szabó Péter,team-szabo
```

| Column     | Required        | Meaning |
|------------|-----------------|---------|
| `username` | this or `email` | GitLab username. The student is added to the repository directly. A leading `@` is ignored. |
| `email`    | this or `username` | Used only when `username` is empty. GitLab sends an invitation, unless you pass `--no-invite`. |
| `name`     | no              | The student's name, used in the repository description. |
| `repo`     | no              | Full repository path for this student. Overrides `--prefix` + username. |

**How repositories are named:** `--prefix` + username, or `--prefix` + the part of the email before the `@`. The name is converted to a valid GitLab path: lowercase, accents removed (`Ádám` → `adam`), and other characters replaced by `-`. If two students would get the same repository name (e.g. `john@a.edu` and `john@b.edu`), the script stops before creating anything. Set the `repo` column for one of them to fix it.

`students.example.csv` contains the example above.

## Running with uv

Set the token in your shell first:

```bash
export GITLAB_TOKEN=glpat-...            # Windows PowerShell: $env:GITLAB_TOKEN="glpat-..."
```

Always start with `--dry-run`. It reads from GitLab and shows what would happen, without changing anything. Run the same command without `--dry-run` to apply it.

### Option A: let uv handle everything (no venv needed)

The script lists its own dependencies in a comment block at the top (the [inline script metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/) standard). `uv run` reads that block, installs the packages into a cached environment, and runs the script:

```bash
uv run provision_student_repos.py \
    --url https://gitlab.example.edu \
    --group nlp-fm/2026-27-1 \
    --students students.csv \
    --prefix hw1- \
    --dry-run
```

The first run installs the packages. Later runs reuse the cached environment and start immediately.

### Option B: a project venv from requirements.txt

```bash
uv venv                                  # creates .venv in the current folder
uv pip install -r requirements.txt       # installs into .venv, no activation needed

uv run python provision_student_repos.py \
    --url https://gitlab.example.edu \
    --group nlp-fm/2026-27-1 \
    --students students.csv \
    --prefix hw1- \
    --dry-run
```

`uv run python ...` finds the `.venv` in the current folder (or a parent folder) on its own. Keep the word `python` in this command. Without it, uv follows Option A and builds its own environment instead of using `.venv`.

If you prefer, you can activate the venv (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) and then run `python provision_student_repos.py ...`.

### Without uv

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python provision_student_repos.py --url ... --group ... --students students.csv --dry-run
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url URL` | `$GITLAB_URL` | GitLab base URL, e.g. `https://gitlab.example.edu`. |
| `--group GROUP` | required | Group path (e.g. `nlp-fm/2026-27-1`) or numeric ID. |
| `--students FILE` | required | Students CSV file. |
| `--prefix PREFIX` | none | Prefix for repository names, e.g. `hw1-`. Not applied to rows with a `repo` value. |
| `--access-level LEVEL` | `developer` | The student's role in their repository: `reporter` (read only), `developer` or `maintainer`. |
| `--expires-at YYYY-MM-DD` | none | Date when student access ends automatically, e.g. the end of the semester. Applies to members and invitations. |
| `--keep-branch-protection` | off | Don't allow Developers to push to the default branch. |
| `--no-invite` | off | Create repositories for email-only students but don't send invitations. Re-run without this flag to invite them later. |
| `--delete-if-no-user` | off | If a username has no GitLab account, delete that student's repository, or don't create it. Repositories with commits beyond the initial README are never deleted. Has no effect on email-only rows. |
| `--dry-run` | off | Show what would happen without changing anything. Also shows which repositories `--delete-if-no-user` would delete. |
| `--no-color` | off | Plain output without colours. Setting the `NO_COLOR` environment variable does the same. |
| `-v`, `--verbose` | off | Debug logging. |

The token is read only from the `GITLAB_TOKEN` environment variable, so it never ends up in your shell history.

**Exit codes:** `0` all students OK · `1` some students failed (see the log; fix the problem and re-run) · `2` setup error (missing token, bad CSV, group not found, …).

## Output

Each line starts with a mark that shows what happened. In a terminal, the lines are also coloured:

| Mark | Colour | Meaning |
|------|--------|---------|
| ✓ | green | Something was changed successfully (repository created, student added or invited, …) |
| • | none | Information, or something that was already in place (e.g. on a re-run) |
| ! | yellow | Warning that needs your attention (dry run, a student in the group, a non-private repository, …) |
| ✗ | red | A student failed, or the run as a whole has failures |

Colours are switched off automatically when the output goes to a file or a pipe. You can also turn them off with `--no-color` or the `NO_COLOR` environment variable. If your terminal can't display the marks, they are replaced by `+`, `-`, `!` and `x`. On Windows, colours need a terminal that supports them, such as Windows Terminal. In the old console window, use `--no-color`.

Example run with `--prefix hw1- --delete-if-no-user` (✓ lines are green, ✗ lines red):

```text
• Signed in as teacher, target group: nlp-fm/2026-27-1
✓ hw1-kovacs.adam: created project https://gitlab.example.edu/nlp-fm/2026-27-1/hw1-kovacs.adam
✓ hw1-kovacs.adam: Developers may now push to main (force push still blocked)
✓ hw1-kovacs.adam: added kovacs.adam as Developer
• hw1-nagy.eva: project already exists
• hw1-nagy.eva: nagy.eva@example.edu already has a pending invitation
✗ hw1-szabo_p: FAILED: no GitLab user with username 'szabo_p'; deleted repository nlp-fm/2026-27-1/hw1-szabo_p
✗ Finished: 2 of 3 students OK; failed: szabo_p
✗ Fix the problems above and re-run; students that are already set up are skipped
```

## Troubleshooting

- **`group '...' not found`**: `--group` was given a display name, or the token's user can't see the group. Use the path or ID (see [step 2](#2-find-the-groups-path-or-id)).
- **`no GitLab user with username '...'`**: the username is misspelled, or the student has no account yet. Fix the username, or leave it empty and fill in `email` so the student gets an invitation instead. The repository is kept unless you pass `--delete-if-no-user`. To clean up repositories created for misspelled usernames, re-run with `--delete-if-no-user --dry-run` first, check the list, then run it again without `--dry-run`. The old (misspelled) usernames have to stay in the CSV for this, because the script only looks at the students listed there.
- **`repository ... was NOT deleted because it contains commits beyond the initial README`**: `--delete-if-no-user` found work in the repository, so it left it alone. Check whether the student renamed their GitLab account. If you are sure the repository can go, delete it by hand in GitLab.
- **`... is scheduled for deletion`**: depending on your GitLab version and settings, a deleted repository may first be marked for deletion and only removed after a waiting period. Restore it in GitLab if you need it, or wait until it's gone before re-creating it.
- **An invitation fails**, e.g. "blocked by domain restrictions": the group or instance only allows invitations to certain email domains. Use the student's GitLab username or an allowed address.
- **Students get an email from GitLab even with `--no-invite`**: students added by username receive GitLab's standard "you have been granted access" notification. The GitLab API has no option to suppress it.
- **A warning says an existing project is not private**: the repository already existed with a different visibility. The script does not change it, so check it by hand.
- **A warning says a student can see ALL student repositories**: that student is a member of the group or a parent group. Remove them from the group, since project membership is all they need.

## Running the tests

The tests simulate the GitLab API, so they need no token or server:

```bash
uv venv
uv pip install -r requirements.txt pytest responses
uv run python -m pytest
```
