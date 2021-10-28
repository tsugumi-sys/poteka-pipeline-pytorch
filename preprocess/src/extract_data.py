def parse_parquet(df, root_dir: str) -> str:
    # df = raw_data["data"].merge(raw_data["target"], how="inner", left_index=True, right_index=True)
    # df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # half_length = int(len(df) / 2)
    # three_quarter_length = half_length + int(half_length / 2)
    # train_df = df[:half_length]
    # test_df = df[half_length:three_quarter_length]
    # valid_df = df[three_quarter_length:]
    # df_list = [test_df, train_df, valid_df]

    # test_filename, train_filename, valid_filename = "test.parquet.gzip", "train.parquet.gzip", "valid.parquet.gzip"
    # filename_list = [test_filename, train_filename, valid_filename]
    filename = "data.parquet.gzip"
    df.to_parquet(
        f"{root_dir}/{filename}",
        engine="pyarrow",
        compression="gzip",
    )

    return filename
