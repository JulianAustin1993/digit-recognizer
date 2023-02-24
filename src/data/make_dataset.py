import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data into cleaned data ready to be
    analyzed.
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")
    train_fp = Path(input_filepath, "train.csv")
    test_fp = Path(input_filepath, "test.csv")  # type: ignore
    trainval_df = pd.read_csv(train_fp, dtype=np.float32)
    seed = int(os.getenv("SEED"))  # type: ignore
    val_split = float(os.getenv("VAL_SPLIT", 0.2)) #type:ignore
    logger.info(f"Splitting train data set into training and validation")
    logger.info(f"proportion training:\t{1-val_split:.3f} ")
    train, val = train_test_split(trainval_df, test_size=val_split, random_state=seed)
    logger.info(f"Training shape: {train.shape}")  # type: ignore
    logger.info(f"Validation shape: {val.shape}")  # type: ignore
    test = pd.read_csv(test_fp, dtype=np.float32)
    logger.info(f"Test shape: {test.shape}")
    output_filepath = Path(output_filepath)
    output_filepath.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing processed data to: {output_filepath}")
    train.to_csv(output_filepath / "train.csv", index=False)  # type: ignore
    val.to_csv(output_filepath / "validation.csv", index=False)  # type: ignore
    test.to_csv(output_filepath / "test.csv", index=False)  # type: ignore
    logger.info("Finished...")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fhandler = logging.FileHandler(filename="mylog.log", mode="a")
    shandler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, handlers=[shandler, fhandler]
    )

    # not used in this stub but often useful for finding various files
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
