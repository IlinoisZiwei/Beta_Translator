"""Resolve saved project artifacts in this checkout, never in the author's checkout."""
import os
from pathlib import Path


def project_path(value, context):
    path, context = Path(value), Path(context).resolve()
    root = next((p for p in (context.parent, *context.parents)
                 if (p / 'beta_translator').is_dir()), None)
    if root is not None:
        # Legacy manifests used absolute paths. Rebase even if that old file exists.
        for index, part in enumerate(path.parts):
            if part in ('outputs', 'videos', 'models', 'data'):
                return root.joinpath(*path.parts[index:])
    return path if path.is_absolute() else context.parent / path


def artifact_paths(manifest_path, manifest):
    base = Path(manifest_path).resolve().parent
    return {(base / p if manifest.get('artifact_path_base') == 'manifest'
             else project_path(p, manifest_path)).resolve(): digest
            for p, digest in manifest.get('artifacts', {}).items()}


def set_artifacts(manifest_path, manifest, updates):
    """Normalize legacy keys before merging, so changed hashes replace old entries."""
    paths = artifact_paths(manifest_path, manifest)
    paths.update({Path(p).resolve(): digest for p, digest in updates.items()})
    manifest['artifacts'] = {os.path.relpath(p, Path(manifest_path).resolve().parent): digest
                             for p, digest in paths.items()}
    manifest['artifact_path_base'] = 'manifest'
