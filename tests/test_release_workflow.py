from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _workflow_text() -> str:
    return RELEASE_WORKFLOW.read_text(encoding="utf-8")


def test_release_workflow_supports_nonpublishing_preflight_builds():
    workflow = _workflow_text()

    assert 'branches:\n      - "codex/*-preflight"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "version:" in workflow
    assert "default: \"v0.2.1-dev\"" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "retention-days: 7" in workflow


def test_release_creation_and_upload_remain_tag_only():
    workflow = _workflow_text()
    tag_condition = "startsWith(github.ref, 'refs/tags/v')"

    assert workflow.count(tag_condition) >= 3
    assert "if: ${{ startsWith(github.ref, 'refs/tags/v') }}" in workflow
    assert "needs.create-release.result == 'skipped'" in workflow
    assert "needs.create-release.result == 'success'" in workflow
    assert "if: ${{ !startsWith(github.ref, 'refs/tags/v') }}" in workflow


def test_release_workflow_uses_one_build_recipe_for_tags_and_preflights():
    workflow = _workflow_text()

    assert workflow.count("packaging/build_release.py --version") == 2
    assert "BUILD_VERSION" in workflow
    assert "github.event.inputs.version" in workflow
    assert "GITHUB_REF_NAME" in workflow
