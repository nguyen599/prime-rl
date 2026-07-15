from datasets import Dataset

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.sft.data import load_sft_dataset


def test_load_local_parquet_file(tmp_path):
    path = tmp_path / "train.parquet"
    Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    ).to_parquet(path)

    dataset = load_sft_dataset(SFTDataConfig(name=str(path), shuffle=False))

    assert len(dataset) == 1
    assert dataset[0]["messages"][1]["content"] == "answer"
