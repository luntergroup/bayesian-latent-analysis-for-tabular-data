"""The command line: `python -m blat <step> ...` (or `blat <step>` once installed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _model_options(p: argparse.ArgumentParser, k_required: bool = True) -> None:
    g = p.add_argument_group("model (override the declaration's \"model\" section)")
    g.add_argument("--k", type=int, help="number of types K (no default)")
    g.add_argument("--alpha-init", type=float)
    g.add_argument("--kappa", type=float)
    g.add_argument("--beta", type=float, help="per-word Dirichlet concentration")
    g.add_argument("--gamma", type=float, nargs=2, metavar=("G0", "G1"))
    g.add_argument("--residual-sd", type=float,
                   help="s_f for continuous variables that declare neither words nor s_f")


def _model_layer(a) -> dict:
    return {"K": a.k, "alpha_init": a.alpha_init, "kappa": a.kappa, "beta": a.beta,
            "gamma": tuple(a.gamma) if a.gamma else None, "residual_sd": a.residual_sd}


def _inputs_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--decl", help="the fitted declaration (.yaml) from `convert`")
    p.add_argument("--data", help="the .preproc.tsv (default: named by the declaration)")
    p.add_argument("--mask", help="a mask file from `convert --make-mask`")
    p.add_argument("--fold", type=int, help="with a fold mask: hide fold n (1..F)")


def cmd_convert(a) -> int:
    from . import convert as C
    from .declaration import Declaration
    from .io import relative

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.decl is None:
        raw = C.read_raw(a.data, a.sep, tuple(a.missing))
        decl = C.draft(raw, a.id)
        decl.source = relative(a.data, out.parent)
        decl.separator = a.sep
        decl.missing = tuple(a.missing)
        path = out.with_name(out.name + ".draft.yaml")
        decl.save(path)
        kinds = {k: len(decl.of_kind(k)) for k in ("categorical", "ordinal", "continuous")}
        print(f"convert: drafted {path}: {kinds}, {len(decl.excluded)} columns excluded")
        for column, reason in decl.excluded.items():
            print(f"  excluded {column}: {reason}")
        print("Check and edit the draft -- types, level order of ordinals (then set "
              "confirmed: true), exclusions -- and run convert again with --decl.")
        return 0
    decl = Declaration.load(a.decl)
    missing = tuple(a.missing) if a.missing_given else decl.missing
    raw = C.read_raw(a.data, a.sep or decl.separator, missing)
    if a.apply:
        if not decl.fitted:
            raise SystemExit("--apply needs a fitted declaration (the .yaml written by "
                             "convert), whose transforms are reused unchanged")
        fitted = decl
    else:
        fitted = C.fit(raw, decl)
    pre = C.apply(raw, fitted)
    pre_path = out.with_name(out.name + ".preproc.tsv")
    C.write_preprocessed(pre, pre_path)
    if not a.apply:
        fitted.source = relative(a.data, out.parent)
        fitted.preprocessed = pre_path.name
        fitted.missing = missing
        fitted.report, table = C.report(raw, pre, fitted)
        from .formats import write_table
        write_table(table, out.with_name(out.name + ".report.tsv"))
        decl_path = out.with_name(out.name + ".yaml")
        fitted.save(decl_path)
        print(f"convert: wrote {decl_path}, {pre_path.name} and {out.name}.report.tsv: "
              f"{len(pre)} records, {len(fitted.names)} variables")
        for w in fitted.report["warnings"]:
            print(f"  warning: {w}")
    else:
        print(f"convert: wrote {pre_path} with the stored transforms ({len(pre)} records)")
    if a.make_mask is not None:
        codes = C.to_codes(pre, fitted)
        mask = C.make_mask(codes, fitted, a.make_mask, np.random.default_rng(a.seed))
        mask_path = out.with_name(out.name + ".mask.tsv")
        C.write_mask(mask, mask_path)
        kind = "fold" if a.make_mask >= 2 else "binary"
        print(f"convert: wrote {kind} mask {mask_path}")
    return 0


def _load_inputs(a):
    from .io import Inputs

    if a.decl is None:
        raise SystemExit("--decl (the fitted declaration) is required")
    return Inputs(a.decl, a.data, a.mask, a.fold)


def cmd_run(a) -> int:
    from .config import ModelConfig, RunConfig, resolve
    from .fit import fit

    inputs = _load_inputs(a)
    cfg = resolve(ModelConfig, inputs.decl.model, _model_layer(a))
    run = resolve(RunConfig, inputs.decl.run,
                  {"seed": a.seed, "alpha_frozen": a.alpha_frozen, "burnin": a.burnin,
                   "iters": a.iters, "thin": a.thin})
    fit(inputs, cfg, run, a.out, ci_draws=a.ci_draws, progress=not a.quiet)
    return 0


def cmd_calibrate(a) -> int:
    from .calibrate import calibrate
    from .config import ModelConfig, resolve
    from .runs import Run

    from_run = Run(a.from_run) if a.from_run else None
    if from_run is not None and a.decl is None:
        inputs = from_run.inputs
        cfg = resolve(ModelConfig, from_run.record["model"], _model_layer(a))
    else:
        inputs = _load_inputs(a)
        cfg = resolve(ModelConfig, inputs.decl.model, _model_layer(a))
    calibrate(inputs, cfg, a.out, rounds=a.rounds, iters=a.iters, burn=a.burn,
              draws=a.draws, seed=a.seed, from_run=from_run, workers=a.workers,
              progress=not a.quiet)
    return 0


def cmd_ti(a) -> int:
    from .config import ModelConfig, as_dict, resolve
    from .encoding import encode
    from .formats import write_table
    from .io import header, output_dir, relative, write_summary
    from .runs import Run
    from .sampler import state_from_parameters
    from .ti import TiConfig, result_record, run_ti

    ti = TiConfig(temperatures=a.temperatures, power=a.power,
                  burn_per_temperature=a.burn_per_temperature,
                  iters_per_temperature=a.iters_per_temperature,
                  initial_burn=a.initial_burn, direction=a.direction, seed=a.seed,
                  sample_alpha=not a.fixed_alpha)
    rng = np.random.default_rng(a.seed)
    start = None
    run = Run(a.from_run) if a.from_run else None
    if run is not None:
        inputs = run.inputs
        cfg = resolve(ModelConfig, run.record["model"], _model_layer(a))
        if cfg.K != run.K:
            raise SystemExit(f"--k {cfg.K} differs from the run's K = {run.K}")
    else:
        if a.direction == "backward":
            raise SystemExit("a backward path starts from a finished run: give --from-run")
        inputs = _load_inputs(a)
        cfg = resolve(ModelConfig, inputs.decl.model, _model_layer(a))
    data = encode(inputs.codes, inputs.decl, cfg, hidden=inputs.hidden)
    if a.direction == "backward":
        alpha, theta, phi = run.last_draw(rng, use_mean=(a.start == "mean"))
        start = state_from_parameters(data, cfg, alpha, theta, phi, rng)
    if not a.quiet:
        print(f"ti: K={cfg.K} {a.direction}, {ti.temperatures} temperatures",
              file=sys.stderr)
    result = run_ti(data, cfg, ti, rng, start=start, progress=not a.quiet)
    out = output_dir(a.out)
    entries, table = result_record(result, ti)
    record = header("ti")
    record.update(K=cfg.K, model=as_dict(cfg), inputs=inputs.describe(out),
                  from_run=relative(run.path, out) if run else None,
                  start_point=a.start if a.direction == "backward" else "prior", **entries)
    arrays = None
    if a.per_temperature:
        write_table(table, out / "per_temperature.tsv")
        arrays = {"samples": result.samples, "t": result.ladder}
    path = write_summary(out, record, arrays)
    if not a.quiet:
        print(f"ti: log p(x|K={cfg.K}) = {result.log_evidence:.1f} +/- "
              f"{result.total_error:.1f}  -> {path}", file=sys.stderr)
    return 0


def cmd_explain(a) -> int:
    from .explain import explain

    explain(a.run, a.out, variables=a.variables, apparent=a.apparent, burn=a.burn,
            draws=a.draws, seed=a.seed, workers=a.workers, progress=not a.quiet)
    return 0


def cmd_impute(a) -> int:
    from .impute import impute

    impute(a.run, a.out, data_path=a.data, mask_path=a.mask, fold=a.fold,
           point_discrete=a.point_discrete, point_continuous=a.point_continuous,
           phi_draws=a.phi_draws, theta_draws=a.theta_draws, burn=a.burn, seed=a.seed,
           intervals=a.intervals, cells=a.cells, progress=not a.quiet)
    return 0


def cmd_simulate(a) -> int:
    from .simulate import simulate

    paths = simulate(a.out, a.k, n_records=a.records, n_categorical=a.categorical,
                     n_ordinal=a.ordinal, n_continuous=a.continuous, kappa=a.kappa,
                     beta=a.beta, residual_sd=a.residual_sd, max_missing=a.max_missing,
                     n_new=a.new_records, new_missing=a.new_missing, seed=a.seed,
                     name=a.name)
    for kind, path in paths.items():
        print(f"simulate: {kind}: {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="blat", description=__doc__)
    sub = top.add_subparsers(dest="step", required=True)

    p = sub.add_parser("convert", help="raw table -> declaration, preprocessed table, mask")
    p.add_argument("data", help="raw .csv or .tsv")
    p.add_argument("--decl", help="a declaration (.yaml); without it a draft is written")
    p.add_argument("--out", required=True,
                   help="output prefix: PREFIX.yaml, PREFIX.preproc.tsv, ...")
    p.add_argument("--id", help="the ID column (draft only)")
    p.add_argument("--sep", help="field separator (default: from the file extension)")
    p.add_argument("--missing", nargs="*", default=["", "NA", "NaN", "nan"],
                   help="tokens that mean missing (default: '' NA NaN nan)")
    p.add_argument("--apply", action="store_true",
                   help="apply a fitted declaration's transforms unchanged (new records)")
    p.add_argument("--make-mask", type=float, metavar="SPEC",
                   help="also write a mask: a fraction in (0,1) for a binary mask, an "
                        "integer F >= 2 for an F-fold mask")
    p.add_argument("--seed", type=int, default=0, help="seed for the mask")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("run", help="fit one chain")
    _inputs_options(p)
    _model_options(p)
    p.add_argument("--out", required=True, help="output directory")
    g = p.add_argument_group("chain")
    g.add_argument("--seed", type=int)
    g.add_argument("--alpha-frozen", type=int, help="iterations with alpha fixed (2000)")
    g.add_argument("--burnin", type=int, help="total burn-in iterations (4000)")
    g.add_argument("--iters", type=int, help="sampling iterations (30000)")
    g.add_argument("--thin", type=int, help="keep every n-th sampling iteration (10)")
    p.add_argument("--ci-draws", type=int, default=500,
                   help="theta draws kept for its credible intervals (default 500)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("calibrate", help="estimate s_f and N_f per continuous variable")
    _inputs_options(p)
    _model_options(p)
    p.add_argument("--from-run", help="start from this run (its K, inputs and phi)")
    p.add_argument("--out", required=True,
                   help="output directory (declaration.yaml, trace.tsv, summary.yaml)")
    p.add_argument("--rounds", type=int, default=4,
                   help="rounds of short chain + estimate of s_f")
    p.add_argument("--iters", type=int, default=1000, help="iterations per short chain")
    p.add_argument("--burn", type=int, default=200, help="burn-in of each record's chain")
    p.add_argument("--draws", type=int, default=200, help="draws of each record's chain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("ti", help="log p(x | K) by thermodynamic integration")
    _inputs_options(p)
    _model_options(p)
    p.add_argument("--from-run", help="a run: its inputs and settings (and, backward, "
                                      "its last draw as the starting point)")
    p.add_argument("--direction", choices=("forward", "backward"), default="forward")
    p.add_argument("--start", choices=("draw", "mean"), default="draw",
                   help="backward: start from the last stored draw (exact) or the "
                        "posterior mean (approximate)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--temperatures", type=int, default=50)
    p.add_argument("--power", type=float, default=5.0)
    p.add_argument("--burn-per-temperature", type=int, default=150)
    p.add_argument("--iters-per-temperature", type=int, default=250)
    p.add_argument("--initial-burn", type=int, default=200)
    p.add_argument("--fixed-alpha", action="store_true",
                   help="hold alpha at alpha_init (the evidence conditional on alpha)")
    p.add_argument("--per-temperature", action="store_true",
                   help="also write per_temperature.tsv, and the samples in arrays.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_ti)

    p = sub.add_parser("explain", help="out-of-sample variance explained per variable")
    p.add_argument("--run", required=True, help="a run's output directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--variables", nargs="*")
    p.add_argument("--apparent", action="store_true", help="also score in sample")
    p.add_argument("--burn", type=int, default=500)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("impute", help="impute missing or masked cells")
    p.add_argument("--run", required=True, help="a run's output directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--data", help="new records: a .preproc.tsv made with convert --apply")
    p.add_argument("--mask", help="impute (and score) the masked cells")
    p.add_argument("--fold", type=int)
    p.add_argument("--point-discrete", choices=("mode", "median", "mean"), default="mode")
    p.add_argument("--point-continuous", choices=("mean", "median", "mode"), default="mean")
    p.add_argument("--phi-draws", type=int, default=20)
    p.add_argument("--theta-draws", type=int, default=10)
    p.add_argument("--burn", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--intervals", action="store_true",
                   help="also write intervals and level probabilities per imputed cell")
    p.add_argument("--cells", action="store_true",
                   help="with a mask, also write cells.tsv: per-cell scoring details")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_impute)

    p = sub.add_parser("simulate", help="synthetic data from the model")
    p.add_argument("--k", type=int, required=True, help="number of true types")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--name", default="example")
    p.add_argument("--records", type=int, default=2500)
    p.add_argument("--categorical", type=int, default=16)
    p.add_argument("--ordinal", type=int, default=7)
    p.add_argument("--continuous", type=int, default=10)
    p.add_argument("--kappa", type=float,
                   help="alpha_k ~ Exp(kappa); default K, so that E[sum alpha] = 1")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--residual-sd", type=float, default=0.2)
    p.add_argument("--max-missing", type=float, default=0.3)
    p.add_argument("--new-records", type=int, default=200)
    p.add_argument("--new-missing", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)
    return top


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.step == "convert":
        given = argv if argv is not None else sys.argv[1:]
        args.missing_given = "--missing" in given
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"blat {args.step}: error: {error}", file=sys.stderr)
        return 2
