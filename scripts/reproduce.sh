#!/usr/bin/env bash
#
# reproduce.sh - recompute the experiments and redraw their figures.
#
# Usage:
#   scripts/reproduce.sh [--benchmark] [--figures-only] [experiment ...]
#
# Arguments:
#   experiment       names of the experiment directories in data/experiments
#                    (default: all)
#   --benchmark      also rerun the benchmark measurements
#   --figures-only   only redraw the figures; the recall figures need runs/
#                    from an earlier full run
#   -h, --help       show this help
#
# Runs every "#   cargo run" line of data/experiments/<name>/*.toml from
# src/compute (with --force), then every "#   uv run" line from src/analysis.
#
# Requirements: rustup (the toolchain of src/compute/rust-toolchain.toml) and
# uv on PATH.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiments_dir="$root/data/experiments"
compute="$root/src/compute"
analysis="$root/src/analysis"

benchmark=false
figures_only=false
selected=()
for arg in "$@"; do
    case "$arg" in
        --benchmark) benchmark=true ;;
        --figures-only) figures_only=true ;;
        -h|--help) sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $arg" >&2; exit 1 ;;
        *)
            if [[ ! -d "$experiments_dir/$arg" ]]; then
                echo "unknown experiment: $arg (available: $(ls "$experiments_dir" | tr '\n' ' '))" >&2
                exit 1
            fi
            selected+=("$arg") ;;
    esac
done
if [[ ${#selected[@]} -eq 0 ]]; then
    while IFS= read -r name; do selected+=("$name"); done < <(ls "$experiments_dir" | sort)
fi

# Prints "<experiment> <command>" for every comment line starting with "#   $1 ".
commands() {
    local prefix="$1" name file
    for name in "${selected[@]}"; do
        for file in "$experiments_dir/$name/config.toml" $(ls "$experiments_dir/$name"/*.toml | grep -v '/config.toml$' | sort); do
            [[ -f "$file" ]] || continue
            tr -d '\r' < "$file" | sed -n "s/^#   \($prefix .*\)$/$name \1/p"
        done
    done
}

run() {
    local directory="$1"; shift
    echo -e "\033[36m[$directory] $*\033[0m"
    (cd "$directory" && "$@" < /dev/null)
}

started=$SECONDS

if [[ "$figures_only" == false ]]; then
    run "$compute" cargo build --release
    while read -r name command; do
        if [[ "$name" == benchmark && "$benchmark" == false ]]; then
            echo -e "\033[33m[skip] benchmark measurements (use --benchmark)\033[0m"
            continue
        fi
        read -r -a words <<< "$command --force"
        run "$compute" "${words[@]}"
    done < <(commands "cargo run")
fi

run "$analysis" uv sync --locked
while read -r name command; do
    read -r -a words <<< "$command"
    run "$analysis" "${words[@]}"
done < <(commands "uv run")

elapsed=$((SECONDS - started))
printf '\033[32mDone in %02d:%02d:%02d. Check the changes with: git status\033[0m\n' \
    $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
