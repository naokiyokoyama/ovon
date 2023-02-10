import csv
import os
from collections.abc import MutableMapping
from typing import Dict, Iterable, Optional, Set


class ObjectCategoryMapping(MutableMapping):

    _mapping: Dict[str, str]

    def __init__(
        self, mapping_file: str, allowed_categories: Optional[Set[str]] = None
    ) -> None:
        self._mapping = self.limit_mapping(
            self.load_mapping(mapping_file), allowed_categories
        )

    @staticmethod
    def load_mapping(mapping_file: str) -> Dict[str, str]:
        mapping = {}
        with open(mapping_file, "r") as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            is_first_row = True
            for row in tsv_reader:
                if is_first_row:
                    is_first_row = False
                    continue
                raw_name = row[1]
                cat_name = row[-1]
                # Override the category name for plant
                if "plant" in raw_name or "flower" in raw_name:
                    cat_name = "plant"
                mapping[raw_name] = cat_name
                raw_name = row[2]
                mapping[raw_name] = cat_name

        return mapping

    @staticmethod
    def limit_mapping(
        mapping: Dict[str, str], allowed_categories: Optional[Set[str]] = None
    ) -> Dict[str, str]:
        if allowed_categories is None:
            return mapping
        return {k: v for k, v in mapping.items() if v in allowed_categories}

    def get_categories(self):
        return set(self._mapping.values())

    def __getitem__(self, key: str):
        k = self._keytransform(key)
        if k in self._mapping:
            return self._mapping[k]
        return None

    def __setitem__(self, key: str, value: str):
        self._mapping[self._keytransform(key)] = value

    def __delitem__(self, key: str):
        del self._mapping[self._keytransform(key)]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def _keytransform(self, key: str):
        return key.lower()


def get_hm3d_semantic_scenes(
    hm3d_dataset_dir: str, splits: Optional[Iterable[str]] = None
) -> Dict[str, Set[str]]:
    if splits is None:
        splits = ["train", "minival", "val"]

    def include_scene(s):
        if not os.path.isdir(s):
            return False
        return len([f for f in os.listdir(s) if "semantic" in f]) > 0

    def get_basis_file(s):
        return [x for x in os.listdir(s) if x.endswith("basis.glb")][0]

    semantic_scenes = {}  # split -> scene file path
    for split in splits:
        split_dir = os.path.join(hm3d_dataset_dir, split)
        all_scenes = [os.path.join(split_dir, s) for s in os.listdir(split_dir)]
        all_scenes = [s for s in all_scenes if include_scene(s)]
        scene_paths = {os.path.join(s, get_basis_file(s)) for s in all_scenes}
        semantic_scenes[split] = scene_paths

    return semantic_scenes


if __name__ == "__main__":
    cat_map = ObjectCategoryMapping(
        mapping_file="data/Mp3d_category_mapping_updated.tsv",
        allowed_categories={
            "chair",
            "bed",
            "toilet",
            "sofa",
            "plant",
            "tv_monitor",
        },
    )
    print(cat_map.get_categories())
    print("category of `armchair`:", cat_map["armchair"])

    s = get_hm3d_semantic_scenes("habitat-sim/data/scene_datasets/hm3d")
    print(s["minival"])
