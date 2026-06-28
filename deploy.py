
import argparse
import subprocess
import sys
import time
import re
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
README = ROOT / "README.md"
RENDER_YAML = ROOT / "render.yaml"
SERVICE_NAME = "mercadolibre-analytics"


#  Helpers 
def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    kwargs = dict(check=check)
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, **kwargs)


def has_cmd(name: str) -> bool:
    return subprocess.run(["which", name], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0


def print_step(n: int, msg: str) -> None:
    print(f"\n\033[1m[{n}]\033[0m {msg}")


def print_ok(msg: str) -> None:
    print(f"  \033[92m✓\033[0m {msg}")


def print_warn(msg: str) -> None:
    print(f"  \033[93m⚠\033[0m {msg}")


def print_err(msg: str) -> None:
    print(f"  \033[91m✗\033[0m {msg}")


# Steps 
def check_prerequisites(dry_run: bool) -> dict:
    print_step(1, "Checking prerequisites")
    tools = {}
    for tool in ["git", "gh", "render", "curl"]:
        tools[tool] = has_cmd(tool)
        status = "found" if tools[tool] else "not found"
        (print_ok if tools[tool] else print_warn)(f"{tool}: {status}")

    if not tools["git"]:
        print_err("git is required. Install from https://git-scm.com")
        sys.exit(1)
    if not tools["gh"]:
        print_warn("gh CLI not found — GitHub push will be manual.")
        print_warn("Install: brew install gh  or  https://cli.github.com")
    if not tools["render"]:
        print_warn("render CLI not found — will open Render dashboard for first deploy.")
        print_warn("Install: https://render.com/docs/cli")
    return tools


def ensure_github_push(dry_run: bool, tools: dict) -> str:
    """Commit any unstaged changes and push to GitHub. Returns the repo URL """
    print_step(2, "Pushing to GitHub")

    if dry_run:
        print_warn("dry-run: skipping git operations")
        return "https://github.com/<your-username>/mercadolibre-analytics"

    # Init repo if needed
    if not (ROOT / ".git").exists():
        run(["git", "init"])
        run(["git", "add", "."])
        run(["git", "commit", "-m", "initial commit"])
        print_ok("Initialized git repo and committed all files")
    else:
        # Stage and commit any new/modified files
        result = run(["git", "status", "--porcelain"], capture=True)
        if result.stdout.strip():
            run(["git", "add", "."])
            run(["git", "commit", "-m", "chore: deploy updates"])
            print_ok("Committed pending changes")
        else:
            print_ok("Working tree clean — nothing to commit")

    # Push
    if tools["gh"]:
        # Check if remote exists
        remote = run(["git", "remote", "get-url", "origin"], capture=True, check=False)
        if remote.returncode != 0:
            run(["gh", "repo", "create", SERVICE_NAME,
                 "--public", "--push", "--source", str(ROOT)])
            print_ok(f"Created and pushed to github.com/<you>/{SERVICE_NAME}")
        else:
            run(["git", "push", "origin", "main"], check=False)
            print_ok(f"Pushed to {remote.stdout.strip()}")
        repo_url = run(["gh", "repo", "view", "--json", "url", "-q", ".url"],
                       capture=True).stdout.strip()
    else:
        repo_url = "https://github.com/<your-username>/" + SERVICE_NAME
        print_warn("Push manually: git push origin main")
        print_warn("Then create repo at: https://github.com/new")

    return repo_url


def deploy_to_render(dry_run: bool, tools: dict, repo_url: str) -> str:
    """Deploy to Render. Returns the live URL.
    Falls back to browser-based setup if Render CLI is unavailable """
    print_step(3, "Deploying to Render")

    if dry_run:
        print_warn("dry-run: skipping Render deploy")
        return f"https://{SERVICE_NAME}.onrender.com"

    if tools["render"]:
        # Check if service already exists
        services = run(["render", "services", "list", "--output", "json"], capture=True, check=False)
        service_exists = SERVICE_NAME in services.stdout

        if service_exists:
            result = run(["render", "deploys", "create", SERVICE_NAME, "--wait"], capture=True, check=False)
            if result.returncode == 0:
                print_ok("Deploy triggered via Render CLI")
            else:
                print_warn(f"render CLI error: {result.stderr.strip()}")
        else:
            print_warn("Service not yet created on Render.")
            print_warn("Creating via render.yaml — browser will open.")
            webbrowser.open(
                f"https://dashboard.render.com/new?repo={repo_url}"
            )
            print_warn("Connect your repo, Render will auto-detect render.yaml.")
            print_warn("After deploy, run: python deploy.py --update-readme <your-url>")
            return f"https://{SERVICE_NAME}.onrender.com"

        live_url = f"https://{SERVICE_NAME}.onrender.com"
    else:
        # No CLI — open browser for manual first deploy
        dashboard_url = f"https://dashboard.render.com/new?repo={repo_url}"
        print_warn("Opening Render dashboard in browser...")
        print_warn("Steps: New -> Web Service -> connect repo -> Deploy")
        print_warn("Render auto-detects render.yaml - no manual config needed.")
        webbrowser.open(dashboard_url)
        live_url = f"https://{SERVICE_NAME}.onrender.com"
        print_warn("\nAfter deploy completes, run:")
        print_warn(" python deploy.py --update-readme https://<your-actual-url>.onrender.com")

    return live_url


def wait_for_live(url: str, dry_run: bool, max_wait: int = 180) -> bool:
    """Poll the URL until it responds 200 or timeout"""
    print_step(4, f"Waiting for deploy to go live: {url}")

    if dry_run:
        print_warn("dry-run: skipping health check")
        return True

    start = time.time()
    attempt = 0
    while time.time() - start < max_wait:
        attempt += 1
        result = run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url], capture=True, check=False)
        code = result.stdout.strip()
        if code == "200":
            elapsed = int(time.time() - start)
            print_ok(f"Live! ({elapsed}s, attempt {attempt})")
            return True
        print(f"  attempt {attempt}: HTTP {code} — waiting 10s...", end="\r")
        time.sleep(10)

    print_warn(f"Timeout after {max_wait}s. Service may still be spinning up.")
    print_warn("Free-tier Render services take up to 3 minutes on first boot.")
    return False


def update_readme(url: str, dry_run: bool) -> None:
    """Replace the placeholder URL in README.md with the live URL."""
    print_step(5, f"Updating README with live URL: {url}")

    content = README.read_text()

    # Patterns to replace, handles both placeholder and previous real URL
    patterns = [
        r"https://mercadolibre-analytics\.onrender\.com",
        r"https://[a-zA-Z0-9-]+\.onrender\.com",]

    new_content = content
    replaced = False
    for pattern in patterns:
        if re.search(pattern, new_content):
            new_content = re.sub(pattern, url, new_content)
            replaced = True
            break

    if not replaced:
        print_warn("No Render URL placeholder found in README.md - nothing to replace.")
        print_warn(f"Manually add this to your README: {url}")
        return

    if dry_run:
        print_warn("dry-run: would write the following to README.md:")
        for line in new_content.split("\n"):
            if url in line or "onrender" in line.lower():
                print(f"  {line}")
        return

    README.write_text(new_content)
    print_ok(f"README.md updated with: {url}")

    # Commit the README change. 
    add_result = run(["git", "add", "README.md"], check=False, capture=True)
    if add_result.returncode != 0:
        print_warn("Could not 'git add' README.md - commit it manually.")
        return
    commit_result = run(["git", "commit", "-m", f"docs: update dashboard URL to {url}"], check=False, capture=True)
    if commit_result.returncode != 0:
        print_warn("Could not commit README.md (nothing to commit, or git author not configured) - commit it manually.")
        return
    push_result = run(["git", "push", "origin", "main"], check=False, capture=True)
    if push_result.returncode != 0:
        print_warn("Committed locally, but 'git push' failed - push manually: git push origin main")
        return
    print_ok("README committed and pushed")


def print_summary(url: str) -> None:
    print("\n")
    print(f"\033[1m  Dashboard live:\033[0m {url}")
    print(" Cold start (free tier): ~30s after 15min inactivity")
    print(" To redeploy: python deploy.py")
    print("\n")


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy MercadoLibre Analytics dashboard to Render")
    parser.add_argument("--dry-run", action="store_true", help="Print all steps without executing them")
    parser.add_argument("--update-readme", metavar="URL", help="Only update README.md with the given live URL (skips deploy)")
    args = parser.parse_args()

    print("\n\033[1mMercadoLibre Analytics - Deploy to Render\033[0m")
    print()

    # Shortcut: only update README
    if args.update_readme:
        update_readme(args.update_readme, dry_run=False)
        print_summary(args.update_readme)
        return

    tools = check_prerequisites(args.dry_run)
    repo_url = ensure_github_push(args.dry_run, tools)
    live_url = deploy_to_render(args.dry_run, tools, repo_url)
    wait_for_live(live_url, args.dry_run)
    update_readme(live_url, args.dry_run)
    print_summary(live_url)


if __name__ == "__main__":
    main()
