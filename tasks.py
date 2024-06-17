from invoke import task


@task
def lint(ctx):
    """Run linters."""
    ctx.run("ruff check . --fix --exit-zero")
    ctx.run("ruff format .")
    ctx.run("mypy src")
