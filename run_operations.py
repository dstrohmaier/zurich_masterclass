import click

from pathlib import Path

from embedding_processing.word2vec import store_reduced_w2v


@click.command()
@click.argument("operation", type=str)
def run_operation(operation: str) -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    match operation:
        case "reduce_w2v":
            store_reduced_w2v(data_dir)
            store_reduced_w2v(data_dir, n_components=3)
        case _:
            raise ValueError(f"Unknown operation {operation}")


if __name__ == "__main__":
    run_operation()
