from utils.general import load_environment
from utils.health import health_check
import typer

load_environment()

app = typer.Typer()


@app.command()
def health():
    """
    Check the health of the environment.
    """

    if not health_check():
        typer.Exit(code=1)


if __name__ == "__main__":
    load_environment()
    app()
