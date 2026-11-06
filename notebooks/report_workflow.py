"""Read-only report views and explicit links between saved motion and Anny.

No fitting or rendering occurs unless the caller explicitly requests rendering.
Solver implementations, result manifests and historical artifacts stay unchanged.
"""
from pathlib import Path
from urllib.parse import quote
import hashlib
import html
import json

import pandas as pd
from IPython.display import HTML, Markdown, display

from beta_translator.body_pipeline import verified_run
from beta_translator.project_io import file_sha256
from notebooks import trial_workflow as wf


def read_json(path):
    return json.loads(Path(path).read_text())


def check_files(folder, hashes):
    for relative, expected in hashes.items():
        if file_sha256(Path(folder) / relative) != expected:
            raise ValueError("Saved artifact changed: " + str(Path(folder) / relative))


def validate_motion(result, config, profile=None, *, full=False):
    """Reject a wrong trial, profile or interval before attaching a report label."""
    settings = result["provenance"]["config"]
    if result["trial"] != config["trial"] or settings["trial_config"] != config:
        raise ValueError("Saved motion uses a different trial or scene configuration.")
    if profile is not None and settings.get("user_profile") != profile:
        raise ValueError("Saved user profile differs. Select the matching saved run or explicitly fit this user.")
    if result["frame_count"] != result["last_frame"] - result["first_frame"] + 1:
        raise ValueError("Saved motion does not describe a continuous native interval.")
    if result["fps"] != config["fps"]:
        raise ValueError("Saved motion has a different native frame rate.")
    if full and (result["first_frame"], result["last_frame"], result["frame_count"]) != (
        0, config["frame_count"] - 1, config["frame_count"]
    ):
        raise ValueError("The full-sequence report requires every native frame; a short clip cannot replace it.")
    return result


def validate_avatar(result, folder):
    """Match source hashes, selected candidate, profile and native time axis."""
    folder = Path(folder)
    avatar = read_json(folder / "summary.json")
    source = Path(result["output_dir"])
    candidate = result["selected_candidate"]
    expected = {
        "source_summary_sha256": file_sha256(source / "summary.json"),
        "source_trajectory_sha256": file_sha256(source / candidate / "trajectory.npz"),
        "source_metrics_sha256": file_sha256(source / candidate / "metrics.csv"),
        "user_profile": result["provenance"]["config"]["user_profile"],
    }
    if any(avatar.get("provenance", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Anny source hashes or user profile do not match the selected motion.")
    for key in ("trial", "height_cm", "first_frame", "last_frame", "frame_count", "fps"):
        if avatar.get(key) != result[key]:
            raise ValueError("Anny native interval or trial differs: " + key)
    if avatar.get("arm_span_cm") != expected["user_profile"]["arm_span_cm"]:
        raise ValueError("Anny arm span differs from the selected motion.")
    if avatar.get("source_candidate") != candidate:
        raise ValueError("Anny uses a different source candidate.")
    check_files(folder, avatar["output_sha256"])
    return avatar


def saved_full(project, config, profile=None, source_run=None):
    project = Path(project)
    if source_run is not None:
        result = verified_run(project / source_run)
    else:
        index = read_json(project / "configs/avatar_results.json")
        avatar_folder = project / index[config["trial"]]["full"]
        result = verified_run(project / read_json(avatar_folder / "summary.json")["source_run"])
        validate_avatar(result, avatar_folder)
    return validate_motion(result, config, profile, full=True)


def saved_comparison(project, config, profile=None):
    project = Path(project)
    index = read_json(project / "configs/personalized_results.json")
    result = verified_run(project / index[config["trial"]])
    validate_motion(result, config, profile)
    if [result["first_frame"], result["last_frame"]] != config["verification_frames"]:
        raise ValueError("Saved comparison does not match the declared short-clip interval.")
    return result


def matching_avatar(project, result):
    """Return only an exact source match; a new full run cannot show an old avatar."""
    project = Path(project)
    expected = file_sha256(Path(result["output_dir"]) / "summary.json")
    index = read_json(project / "configs/avatar_results.json")
    indexed = [project / value for value in index[result["trial"]].values()]
    folders = list(dict.fromkeys(indexed + sorted((project / "outputs" / result["trial"] / "avatars").glob("*"))))
    for folder in folders:
        if not (folder / "summary.json").is_file():
            continue
        candidate = read_json(folder / "summary.json")
        if candidate.get("provenance", {}).get("source_summary_sha256") == expected:
            validate_avatar(result, folder)
            return folder
    return None


def result_identity(result, role):
    settings = result["provenance"]["config"]
    profile = settings.get("user_profile", {})
    return {
        "Role": role,
        "Frames": f'{result["first_frame"]}–{result["last_frame"]}',
        "Seconds": round(result["frame_count"] / result["fps"], 2),
        "Height / span (cm)": f'{result["height_cm"]:g} / {profile.get("arm_span_cm", "unrecorded")}',
        "Iteration budget": settings["iterations"],
        "Active candidates": ", ".join(row["candidate"] for row in result["ranking"]),
    }


def details(label, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    display(HTML(f"<details><summary>{html.escape(label)}</summary><pre>{html.escape(text)}</pre></details>"))


def artifact_links(project, paths, labels=None):
    project = Path(project)
    entries = []
    for i, path in enumerate(paths):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        label = labels[i] if labels else path.name
        relative = quote(path.relative_to(project).as_posix())
        entries.append(f"[{label}]({relative})")
    display(Markdown(" · ".join(entries)))


def show_motion(project, result, *, details_enabled=False):
    out = Path(result["output_dir"])
    selected = next(row for row in result["ranking"] if row["candidate"] == result["selected_candidate"])
    display(pd.DataFrame([result_identity(result, "Full-sequence diagnostic")]))
    display(Markdown(
        f'**Status:** `{result["selection_status"]}`. **Solver converged:** {selected["converged"]}. '
        f'**Frames with at least one ranked failure:** {selected["total_failed_frames"]}/{result["frame_count"]}. '
        f'**Physical validation:** {result["physical_validation"]}.'
    ))
    counts = {
        "Contact": "contact_failure_frames", "Support": "support_failure_frames",
        "Shoulder envelope": "shoulder_envelope_failure_frames", "Self-collision": "self_collision_failure_frames",
        "Joint-moment capacity": "torque_capacity_failure_frames", "Fatigue": "fatigue_failure_frames",
        "Dynamic balance": "dynamic_balance_failure_frames", "Departing contact load": "departing_force_failure_frames",
    }
    display(pd.DataFrame([
        {"Check": label, "Failed frames": selected[key], "Total frames": result["frame_count"]}
        for label, key in counts.items() if key in selected
    ]))
    if len(result["ranking"]) == 1:
        display(Markdown(f'This saved full run tested one active candidate (`{result["selected_candidate"]}`); it does not establish a winner over untested full-sequence alternatives.'))
    display(Markdown("Counts overlap and must not be added. Contact/support checks use assumed geometry and capacity. See the matching full-sequence videos in Section 9."))
    artifact_links(project, [out / "summary.json", out / "candidate_comparison.csv", out / result["selected_candidate"] / "constraints.csv"],
                   ["Run and provenance", "Candidate metrics", "Complete constraint audit"])
    if details_enabled:
        display(pd.read_csv(out / result["selected_candidate"] / "constraints.csv"))
        display(pd.read_csv(out / result["selected_candidate"] / "metrics.csv").head(8))


def saved_presentation(result, *, render=False):
    """Reuse a verified presentation; missing figures never trigger implicit work."""
    if render:
        from beta_translator.body_render import render_run
        return render_run(Path(result["output_dir"]))
    out = Path(result["output_dir"])
    digest = file_sha256(out / "summary.json")
    from beta_translator import body_render
    renderer = Path(body_render.__file__)
    identity = {"summary": digest, "renderer": file_sha256(renderer)}
    preferred = out / ("presentation_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10])
    folders = list(dict.fromkeys([preferred, *sorted(out.glob("presentation_*"))]))
    for folder in folders:
        if not (folder / "manifest.json").is_file():
            continue
        manifest = read_json(folder / "manifest.json")
        if manifest.get("provenance", {}).get("summary") == digest:
            check_files(folder, manifest["output_sha256"])
            return folder
    return None


def show_comparison(project, result, *, show_video=False, render=False, details_enabled=False):
    display(pd.DataFrame([result_identity(result, "Short-clip comparison only")]))
    rows = [*result["ranking"], *result.get("controls", [])]
    table = pd.DataFrame(rows)
    columns = ["candidate", "eligible_for_selection", "converged", "total_failed_frames", "max_contact_gap_cm", "selection_status"]
    display(table[[column for column in columns if column in table]])
    display(Markdown("The anatomical control shares this clip and user profile, but uses earlier objectives and cannot win selection. This comparison does not rank full-sequence alternatives. Failures remain visible for all candidates."))
    presentation = saved_presentation(result, render=render)
    if presentation is not None:
        artifact_links(project, [presentation / "candidate_comparison_3d.mp4", presentation / "selected_plan_3d.mp4", presentation / "metrics.png"],
                       ["Short-clip comparison video", "Short-clip selected video", "Short-clip metrics"])
        if show_video:
            wf.video(project, presentation / "candidate_comparison_3d.mp4")
    else:
        display(Markdown("No saved presentation matches this run. Metrics are available; video rendering requires `RENDER_COMPARISON=True`."))
    if details_enabled:
        display(table)
    artifact_links(project, [Path(result["output_dir"]) / "summary.json"], ["Short-clip run and provenance"])


def show_avatar(project, result, folder):
    summary = validate_avatar(result, folder)
    display(Markdown(
        f'**Exact source match:** `{Path(result["output_dir"]).name}` / `{result["selected_candidate"]}`. '
        f'Frames {summary["first_frame"]}–{summary["last_frame"]} at {summary["fps"]:g} FPS. '
        f'**Mesh-to-solver mapping passed:** {summary["mapping_pass"]}. '
        f'**Physical validation:** {summary["physical_validation"]}.'
    ))
    display(pd.DataFrame([
        {"Measure": "Maximum mesh-to-solver mapping drift (cm)", "Value": summary["max_patch_drift_cm"]},
        {"Measure": "Frames with mesh-to-hold gap > 0.1 cm", "Value": summary["mesh_contact_failed_frames"]},
        {"Measure": "Frames with source constraint failure", "Value": summary["source_failed_frames"]},
        {"Measure": "Maximum assumed-wall penetration (cm)", "Value": summary["max_wall_penetration_cm"]},
    ]))
    for filename in ("avatar_comparison_3d.mp4", "avatar_source_overlay.mp4"):
        wf.video(project, Path(folder) / filename)
    artifact_links(project, [Path(folder) / "summary.json", Path(folder) / "surface_metrics.csv"],
                   ["Avatar provenance and checks", "Per-frame surface audit"])


def historical_stage(project, item, *, expand=False):
    """Keep historical artifacts discoverable without replaying every video."""
    check_files(project, item["sha256"])
    if expand:
        wf.saved_stage(project, item)
        return
    display(Markdown(f'**Recorded scope:** {item["scope"]}. {item.get("note", "")}'))
    for key, label in (("tables", "Table"), ("images", "Figure"), ("videos", "Video"), ("links", "Reference")):
        paths = item.get(key, [])
        if paths:
            artifact_links(project, [Path(project) / path for path in paths], [f"{label} {i + 1}: {Path(path).name}" for i, path in enumerate(paths)])
    if item.get("summary"):
        artifact_links(project, [Path(project) / item["summary"]], ["Historical run and provenance"])
