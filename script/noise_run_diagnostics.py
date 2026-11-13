"""Command-line entrypoint for saved-run diagnostics."""

from _bootstrap import bootstrap

bootstrap()

from diagnostics.noise_run_diagnostics import main


if __name__ == "__main__":
    main()
