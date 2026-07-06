# Pluto environment template

This `Project.toml` / `Manifest.toml` pair is a **known-good, pinned package
environment** for the `POMDPs.jl` ecosystem used throughout this course
(`POMDPs`, `QuickPOMDPs`, `POMDPModelTools`, `POMDPPolicies`, `BeliefUpdaters`,
`Parameters`, `Plots`, `PlutoUI`, plus the extra solvers used in
`2-POMDPs.jl`/`HandsOnPOMDP.jl`: `QMDP`, `FIB`, `PointBasedValueIteration`).

It exists because the versions of `Plots`/`StatsBase`/`GR` that get resolved
"fresh" today are **not** compatible with this old (2021) POMDPs.jl stack on
any current Julia release — newer `Plots` needs a `GR` that requires Julia
syntax this stack can't have, and letting `StatsBase` float free (on Julia
1.10+) gets pinned down to an old release that calls a `Base.floatrange`
signature that no longer exists in modern Julia. There's a second, unrelated
trap: Pluto itself only ships a modern, bug-free release for Julia 1.10+ —
older Julia (1.9 and below) gets stuck on an old Pluto with a broken
exception-formatting code path (`parentmodule(::Method)` crashes whenever
Pluto tries to display an error).

The fix used everywhere in this repo is: Julia **1.10** + `Plots` pinned to
**1.20.0** + `StatsBase` pinned to whatever version actually resolves for
this package set (currently `0.33.21`) — both compat-locked, so neither can
silently upgrade/downgrade again on a future `Pkg.add`.

## Using this for a new notebook

If you're writing a new Pluto notebook that uses this same package family
(anything in `Project.toml` here), you don't need to `Pkg.add` anything by
hand and risk hitting the same crash. Just ask to have this environment
embedded into your notebook file, or do it yourself from a Julia REPL:

```julia
import Pluto
Pluto.PkgUtils.write_dir_to_nb(
    "notebooks/pluto-environment-template",
    "notebooks/YourNewNotebook.jl",
)
```

This copies this exact `Project.toml`/`Manifest.toml` into your notebook file
(Pluto's native per-notebook package format), so opening it in Pluto never
re-resolves packages from scratch.

If your notebook needs an extra package not listed here, just `Pkg.add` it —
Pluto's package manager respects the existing `Plots`/`StatsBase` compat
locks, so it won't silently re-resolve them to an incompatible version. That
re-resolve (because a new/removed package changed the dependency graph with
no compat lock in place) is exactly what broke `HandsOnMDP` the first time
around.

For a **new notebook**, prefer duplicating
[`notebooks/HandsOn-starter-template.jl`](../HandsOn-starter-template.jl)
instead of using this folder directly — it's this exact environment already
embedded in a blank notebook with the common imports in place.
