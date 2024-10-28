from pipeline.utils import load_environment, health_check
import typer

load_environment()

app = typer.Typer()


@app.command()
def health():
    """
    Check the health of the environment.
    """

    passed, _ = health_check()
    if not passed:
        typer.Exit(code=1)


if __name__ == "__main__":
    load_environment()
    app()
