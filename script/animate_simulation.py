"""Command-line entrypoint for simulation animation."""

from _bootstrap import bootstrap

bootstrap()

from visualization.animate_simulation import main


if __name__ == "__main__":
    main()
