#!/usr/bin/env python3
"""Post-process ANTsNetCT cortical thickness inputs.

This script recomputes ANTs KellyKapowski thickness from the ANTsNetCT
segmentation and posterior probability maps. User provides the derivatives
root, subject, and session, and the script discovers matching ANTsNetCT
outputs beneath that location. If contusion-mask subtraction is requested,
the user must also provide either ``--contusion-mask`` or ``--bids-dir`` so
the mask can be found.

Assumptions
-----------
The script assumes the ANTsNetCT derivative layout and filenames used by the
official package and by this repository's submit wrappers:

    <derivatives-dir>/<antsnetct-dataset>/<sub>/<ses>/anat/
      <prefix>_dseg.nii.gz
      <prefix>_label-WM_probseg.nii.gz
      <prefix>_label-SGM_probseg.nii.gz
      <prefix>_label-CGM_probseg.nii.gz

where ``<prefix>`` ends in ``_seg-antsnetct``. By default
``<antsnetct-dataset>`` is ``output_longi``.

The ANTsNetCT labels are assumed to be:
    WM = 2, CGM = 8, SGM = 9

Post-processing follows the legacy TRACK workflow: SGM is relabeled as WM in
the discrete segmentation, and the WM posterior supplied to KellyKapowski is
WM + SGM. If a contusion mask is enabled, it is binarized, resampled to the
ANTsNetCT segmentation grid, optionally dilated there, then removed from the
segmentation and GM/WM posteriors before KellyKapowski is run.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ANTSNETCT_DATASET = "output_longi"
DEFAULT_OUTPUT_DESC = "thicknessPostproc"
DEFAULT_CONTUSION_OUTPUT_DESC = "thicknessPostprocNoContusion"
DEFAULT_PREPARED_MASK_DESC = "contusionMaskPostproc"

GM_LABEL = 8
WM_LABEL = 2
SGM_LABEL = 9

DEFAULT_CONTUSION_PATTERNS = (
    "*_label-contusion*.nii.gz",
    "*_label-lesion_roi.nii.gz",
    "*_label-lesion*.nii.gz",
)


@dataclass(frozen=True)
class AntsNetCtTarget:
    """File group needed to run KellyKapowski for one ANTsNetCT prefix."""

    prefix: str
    anat_dir: str
    dseg_path: str
    wm_probseg_path: str
    sgm_probseg_path: str
    cgm_probseg_path: str
    output_path: str
    contusion_output_path: str
    prepared_contusion_mask_path: str

    @property
    def required_paths(self) -> Tuple[str, str, str, str]:
        return (
            self.dseg_path,
            self.wm_probseg_path,
            self.sgm_probseg_path,
            self.cgm_probseg_path,
        )


def normalize_bids_label(value: str, prefix: str) -> str:
    """Return a BIDS-like label with the requested prefix."""

    return value if value.startswith(prefix) else f"{prefix}{value}"


def bool_from_int(value: int) -> bool:
    """Convert legacy 0/1 argparse values to bool."""

    return value == 1


def positive_or_zero_int(value: str) -> int:
    """Argparse type for non-negative integer options."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def antsnetct_anat_dir(derivatives_dir: str, dataset: str, sub: str, ses: str) -> str:
    """Return the expected subject/session anat directory."""

    return os.path.join(derivatives_dir, dataset, sub, ses, "anat")


def strip_nii_gz_suffix(path: str) -> str:
    """Return a filename stem while preserving ordinary dots in the prefix."""

    filename = os.path.basename(path)
    suffix = ".nii.gz"
    return filename[: -len(suffix)] if filename.endswith(suffix) else os.path.splitext(filename)[0]


def discover_prefixes(anat_dir: str, requested_prefix: Optional[str]) -> List[str]:
    """Find ANTsNetCT filename prefixes in an anat directory.

    ``requested_prefix`` may be either a basename prefix or a full path to a
    ``*_dseg.nii.gz`` file. Discovery is otherwise based on the ANTsNetCT
    ``*_seg-antsnetct_dseg.nii.gz`` suffix rather than a project-specific
    acquisition label.
    """

    if requested_prefix is not None:
        prefix = strip_nii_gz_suffix(requested_prefix)
        if prefix.endswith("_dseg"):
            prefix = prefix[: -len("_dseg")]
        return [prefix]

    pattern = os.path.join(anat_dir, "*_seg-antsnetct_dseg.nii.gz")
    prefixes: List[str] = []
    for dseg_path in sorted(glob.glob(pattern)):
        filename = os.path.basename(dseg_path)
        prefixes.append(filename[: -len("_dseg.nii.gz")])

    return deduplicate(prefixes)


def deduplicate(values: Iterable[str]) -> List[str]:
    """Return values in their first-seen order with duplicates removed."""

    seen = set()
    unique: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def target_for_prefix(
    anat_dir: str,
    prefix: str,
    output_desc: str,
    contusion_output_desc: str,
    prepared_mask_desc: str,
) -> AntsNetCtTarget:
    """Build all file paths associated with one ANTsNetCT prefix."""

    return AntsNetCtTarget(
        prefix=prefix,
        anat_dir=anat_dir,
        dseg_path=os.path.join(anat_dir, f"{prefix}_dseg.nii.gz"),
        wm_probseg_path=os.path.join(anat_dir, f"{prefix}_label-WM_probseg.nii.gz"),
        sgm_probseg_path=os.path.join(anat_dir, f"{prefix}_label-SGM_probseg.nii.gz"),
        cgm_probseg_path=os.path.join(anat_dir, f"{prefix}_label-CGM_probseg.nii.gz"),
        output_path=os.path.join(anat_dir, f"{prefix}_desc-{output_desc}.nii.gz"),
        contusion_output_path=os.path.join(anat_dir, f"{prefix}_desc-{contusion_output_desc}.nii.gz"),
        prepared_contusion_mask_path=os.path.join(anat_dir, f"{prefix}_desc-{prepared_mask_desc}_mask.nii.gz"),
    )


def missing_required_paths(target: AntsNetCtTarget) -> List[str]:
    """Return required ANTsNetCT inputs that are absent on disk."""

    return [path for path in target.required_paths if not os.path.exists(path)]


def discover_contusion_masks(
    bids_dir: str,
    sub: str,
    ses: str,
    patterns: Sequence[str],
) -> List[str]:
    """Find candidate BIDS-space contusion masks for a subject/session."""

    anat_dir = os.path.join(bids_dir, sub, ses, "anat")
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(sorted(glob.glob(os.path.join(anat_dir, pattern))))
    return deduplicate(matches)


def choose_contusion_mask(
    explicit_mask: Optional[str],
    bids_dir: Optional[str],
    sub: str,
    ses: str,
    patterns: Sequence[str],
) -> Tuple[Optional[str], List[str]]:
    """Resolve the contusion mask path and return all discovered candidates."""

    if explicit_mask is not None:
        return explicit_mask, [explicit_mask]

    if bids_dir is None:
        return None, []

    candidates = discover_contusion_masks(bids_dir, sub, ses, patterns)
    return (candidates[0] if candidates else None), candidates


def prepare_contusion_mask(
    ants: Any,
    mask_path: str,
    reference_image: Any,
    dilation_radius: int,
    dilation_shape: str,
) -> Any:
    """Load, binarize, resample, and optionally dilate a contusion mask.

    Dilation is performed after resampling so the radius is in voxels of the
    ANTsNetCT output grid. The returned mask is binary-valued on that grid.
    """

    mask = ants.threshold_image(ants.image_read(mask_path), low_thresh=0.1, inval=1, outval=0)
    mask = ants.resample_image_to_target(mask, reference_image, interp_type="nearestNeighbor")
    mask = ants.threshold_image(mask, low_thresh=0.5, inval=1, outval=0)

    if dilation_radius > 0:
        mask = ants.morphology(
            mask,
            operation="dilate",
            radius=dilation_radius,
            mtype="binary",
            value=1,
            shape=dilation_shape,
        )
        mask = ants.threshold_image(mask, low_thresh=0.5, inval=1, outval=0)

    return mask


def subtract_mask(image: Any, mask: Optional[Any]) -> Any:
    """Remove masked voxels from an ANTs image while preserving image metadata."""

    if mask is None:
        return image
    return image - (image * mask)


def run_kelly_kapowski(
    ants: Any,
    target: AntsNetCtTarget,
    contusion_mask: Optional[Any],
    kk_iterations: int,
    kk_r: float,
    kk_m: float,
) -> Any:
    """Run KellyKapowski thickness after ANTsNetCT-specific label handling."""

    kk_seg = ants.image_read(target.dseg_path)

    # KellyKapowski expects a GM label and WM label. ANTsNetCT keeps SGM as a
    # separate class, so this post-processing treats SGM as WM for thickness.
    kk_seg[kk_seg == SGM_LABEL] = WM_LABEL
    kk_seg = subtract_mask(kk_seg, contusion_mask)

    wm_posterior = ants.image_read(target.wm_probseg_path)
    sgm_posterior = ants.image_read(target.sgm_probseg_path)
    cgm_posterior = ants.image_read(target.cgm_probseg_path)

    kk_wm_posterior = subtract_mask(wm_posterior + sgm_posterior, contusion_mask)
    kk_gm_posterior = subtract_mask(cgm_posterior, contusion_mask)

    return ants.kelly_kapowski(
        s=kk_seg,
        g=kk_gm_posterior,
        w=kk_wm_posterior,
        its=kk_iterations,
        r=kk_r,
        m=kk_m,
        gm_label=GM_LABEL,
        wm_label=WM_LABEL,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface used by ``main`` and ``--help``."""

    parser = argparse.ArgumentParser(
        description="Recompute KellyKapowski thickness from ANTsNetCT outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            f"""\
            Examples:
              python antsnetct_postproc_thickness.py \\
                --derivatives-dir /path/to/derivatives --sub sub-001 --ses ses-2WK

              python antsnetct_postproc_thickness.py \\
                --derivatives-dir /path/to/derivatives --sub 001 --ses 2WK \\
                --bids-dir /path/to/bids --use-contusion-mask 1 \\
                --contusion-dilation-radius 2

              python antsnetct_postproc_thickness.py \\
                --derivatives-dir /path/to/derivatives --sub sub-001 --ses ses-2WK \\
                --contusion-mask /path/to/sub-001_ses-2WK_label-contusion_roi.nii.gz

            Notes:
              * Required inputs are --derivatives-dir, --sub, and --ses.
              * For contusion-mask discovery, also provide --bids-dir unless
                supplying --contusion-mask directly.
              * Default ANTsNetCT dataset: {DEFAULT_ANTSNETCT_DATASET}
              * Default primary output desc: {DEFAULT_OUTPUT_DESC}
              * --use-lesion-mask is retained as a deprecated alias for
                --use-contusion-mask to keep older submit wrappers working.
            """
        ),
    )
    parser.add_argument(
        "--derivatives-dir",
        required=True,
        help="Derivatives root that contains the ANTsNetCT dataset directory.",
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. sub-141048 or 141048.")
    parser.add_argument("--ses", required=True, help="Session ID, e.g. ses-2WK or 2WK.")
    parser.add_argument(
        "--antsnetct-dataset",
        default=DEFAULT_ANTSNETCT_DATASET,
        help="Dataset directory inside --derivatives-dir, usually output_longi or output_cross.",
    )
    parser.add_argument(
        "--prefix",
        help=(
            "Optional ANTsNetCT filename prefix to process. By default all "
            "*_seg-antsnetct_dseg.nii.gz prefixes in the subject/session anat "
            "directory are processed."
        ),
    )
    parser.add_argument(
        "--output-desc",
        default=DEFAULT_OUTPUT_DESC,
        help="BIDS desc value for the primary thickness output.",
    )
    parser.add_argument(
        "--contusion-output-desc",
        default=DEFAULT_CONTUSION_OUTPUT_DESC,
        help="Additional BIDS desc value written when a contusion mask is applied.",
    )
    parser.add_argument(
        "--bids-dir",
        help=(
            "BIDS root used to discover contusion masks when --contusion-mask "
            "is not provided. Required when --use-contusion-mask 1 is set "
            "without --contusion-mask."
        ),
    )
    parser.add_argument(
        "--contusion-mask",
        help="Explicit contusion/lesion mask path. Providing this enables contusion masking.",
    )
    parser.add_argument(
        "--contusion-mask-pattern",
        action="append",
        help=(
            "Glob pattern searched under <bids-dir>/<sub>/<ses>/anat. May be "
            "specified more than once. Defaults cover common contusion/lesion "
            "BIDS names."
        ),
    )
    parser.add_argument(
        "--use-contusion-mask",
        "--use-lesion-mask",
        dest="use_contusion_mask",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, subtract a contusion mask before KellyKapowski. Legacy alias: --use-lesion-mask.",
    )
    parser.add_argument(
        "--require-contusion-mask",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, fail when contusion masking is requested but no mask is found.",
    )
    parser.add_argument(
        "--contusion-dilation-radius",
        type=positive_or_zero_int,
        default=0,
        help=(
            "Binary morphology dilation radius in ANTsNetCT target-grid voxels. "
            "Use 0 to disable dilation."
        ),
    )
    parser.add_argument(
        "--contusion-dilation-shape",
        choices=["ball", "box", "cross", "annulus", "polygon"],
        default="ball",
        help="Structuring-element shape for contusion-mask dilation.",
    )
    parser.add_argument(
        "--write-prepared-contusion-mask",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, save the resampled/dilated contusion mask beside the thickness output.",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, overwrite existing primary post-processing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, print discovered inputs and planned outputs without loading ANTs.",
    )
    parser.add_argument(
        "--kk-iterations",
        type=positive_or_zero_int,
        default=45,
        help="KellyKapowski iterations passed as ants.kelly_kapowski(..., its=...).",
    )
    parser.add_argument(
        "--kk-r",
        type=float,
        default=0.025,
        help="KellyKapowski gradient-step parameter passed as r.",
    )
    parser.add_argument(
        "--kk-m",
        type=float,
        default=1.5,
        help="KellyKapowski smoothing/regularization parameter passed as m.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    return build_arg_parser().parse_args(argv)


def print_target_plan(targets: Sequence[AntsNetCtTarget], contusion_mask_path: Optional[str]) -> None:
    """Print the dry-run processing plan."""

    print("Discovered ANTsNetCT targets:")
    for target in targets:
        print(f"  prefix: {target.prefix}")
        print(f"    dseg: {target.dseg_path}")
        print(f"    WM:   {target.wm_probseg_path}")
        print(f"    SGM:  {target.sgm_probseg_path}")
        print(f"    CGM:  {target.cgm_probseg_path}")
        print(f"    out:  {target.output_path}")
        if contusion_mask_path is not None:
            print(f"    contusion out: {target.contusion_output_path}")
            print(f"    prepared mask: {target.prepared_contusion_mask_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ANTsNetCT post-processing workflow."""

    args = parse_args(argv)

    sub = normalize_bids_label(args.sub, "sub-")
    ses = normalize_bids_label(args.ses, "ses-")
    anat_dir = antsnetct_anat_dir(args.derivatives_dir, args.antsnetct_dataset, sub, ses)
    prefixes = discover_prefixes(anat_dir, args.prefix)

    if not prefixes:
        print(
            f"ERROR: no ANTsNetCT dseg files found in {anat_dir} "
            "(expected '*_seg-antsnetct_dseg.nii.gz').",
            file=sys.stderr,
        )
        return 1

    targets = [
        target_for_prefix(
            anat_dir=anat_dir,
            prefix=prefix,
            output_desc=args.output_desc,
            contusion_output_desc=args.contusion_output_desc,
            prepared_mask_desc=DEFAULT_PREPARED_MASK_DESC,
        )
        for prefix in prefixes
    ]

    use_contusion_mask = bool_from_int(args.use_contusion_mask) or args.contusion_mask is not None
    contusion_patterns = tuple(args.contusion_mask_pattern or DEFAULT_CONTUSION_PATTERNS)
    contusion_mask_path: Optional[str] = None
    contusion_candidates: List[str] = []

    if use_contusion_mask:
        if args.contusion_mask is None and args.bids_dir is None:
            print(
                "ERROR: --bids-dir is required to discover a contusion mask "
                "when --use-contusion-mask 1 is set without --contusion-mask.",
                file=sys.stderr,
            )
            return 1

        contusion_mask_path, contusion_candidates = choose_contusion_mask(
            explicit_mask=args.contusion_mask,
            bids_dir=args.bids_dir,
            sub=sub,
            ses=ses,
            patterns=contusion_patterns,
        )

        if args.contusion_mask is not None and not os.path.exists(args.contusion_mask):
            print(f"ERROR: explicit contusion mask does not exist: {args.contusion_mask}", file=sys.stderr)
            return 1

        if contusion_mask_path is None:
            message = (
                f"No contusion mask found for {sub} {ses}. "
                "Provide --contusion-mask or --bids-dir with matching --contusion-mask-pattern."
            )
            if bool_from_int(args.require_contusion_mask):
                print(f"ERROR: {message}", file=sys.stderr)
                return 1
            print(f"WARNING: {message} Continuing without contusion subtraction.", file=sys.stderr)
            use_contusion_mask = False
        elif len(contusion_candidates) > 1:
            print(f"WARNING: found {len(contusion_candidates)} contusion-mask candidates; using first:")
            print(f"  {contusion_mask_path}")

    if bool_from_int(args.dry_run):
        print(f"Subject/session: {sub} {ses}")
        print(f"ANTsNetCT anat dir: {anat_dir}")
        if use_contusion_mask:
            print(f"Contusion mask: {contusion_mask_path}")
            print(f"Contusion dilation radius: {args.contusion_dilation_radius}")
            print(f"Contusion dilation shape: {args.contusion_dilation_shape}")
        else:
            print("Contusion mask: disabled")
        print_target_plan(targets, contusion_mask_path if use_contusion_mask else None)
        return 0

    try:
        import ants
    except ImportError as exc:
        print(
            "ERROR: could not import ants. Run this script in an ANTsPy/ANTsNetCT environment.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    processed = 0
    skipped = 0
    failures = 0

    for target in targets:
        print(f"\n[{sub} {ses}] {target.prefix}")

        if not bool_from_int(args.overwrite) and os.path.exists(target.output_path):
            print(f"Skipping existing output: {target.output_path}")
            skipped += 1
            continue

        missing = missing_required_paths(target)
        if missing:
            print(f"ERROR: missing required files for prefix {target.prefix}:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            failures += 1
            continue

        contusion_mask = None
        if use_contusion_mask and contusion_mask_path is not None:
            print(f"Using contusion mask: {contusion_mask_path}")
            print(
                "Preparing contusion mask "
                f"(dilation radius={args.contusion_dilation_radius}, shape={args.contusion_dilation_shape})"
            )
            reference = ants.image_read(target.dseg_path)
            contusion_mask = prepare_contusion_mask(
                ants=ants,
                mask_path=contusion_mask_path,
                reference_image=reference,
                dilation_radius=args.contusion_dilation_radius,
                dilation_shape=args.contusion_dilation_shape,
            )
            if bool_from_int(args.write_prepared_contusion_mask):
                ants.image_write(contusion_mask, filename=target.prepared_contusion_mask_path)
                print(f"Wrote prepared contusion mask: {target.prepared_contusion_mask_path}")

        try:
            kk = run_kelly_kapowski(
                ants=ants,
                target=target,
                contusion_mask=contusion_mask,
                kk_iterations=args.kk_iterations,
                kk_r=args.kk_r,
                kk_m=args.kk_m,
            )
            ants.image_write(kk, filename=target.output_path)
            print(f"Wrote primary thickness output: {target.output_path}")

            if contusion_mask is not None:
                ants.image_write(kk, filename=target.contusion_output_path)
                print(f"Wrote contusion-specific thickness output: {target.contusion_output_path}")

            processed += 1
        except Exception as exc:  # noqa: BLE001 - report target-level failures in batch runs.
            failures += 1
            print(f"ERROR: failed while processing {target.prefix}: {exc}", file=sys.stderr)

    print(f"\nSummary: processed={processed}, skipped={skipped}, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
