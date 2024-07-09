import os
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sympy.printing.pretty.stringpict import prettyForm

plt.switch_backend("tkagg")

__all__ = ["generate_annotation", "split_train_test"]


def generate_annotation(root_path: str) -> None:
    columns = ["img", "fabric", "class"]
    class_names = ["level_" + str(i) for i in range(1, 6)]
    #########################################################################################

    df = pd.DataFrame(columns=columns)

    for path, name, filename in os.walk(root_path):
        if path.split(os.sep)[-1] in class_names:
            prefix_path = f"{os.sep}".join(path.split(os.sep)[-2:])
            class_name = path.split(os.sep)[-1]
            fabric = path.split(os.sep)[-2].split(" ")[0]

            for img_name in filename:
                new_reccord = [os.path.join(prefix_path, img_name), fabric, class_name]
                df = df._append({col: data for col, data in zip(columns, new_reccord)}, ignore_index=True)
    df.to_csv(path_or_buf=os.path.join(root_path, "annotation.csv"), index=False)
    return None


def split_train_test(root_path: str,
                     annotation_file: str,
                     train_size: float = .8,
                     data_types = ["train", "test"],
                     ) -> None:
    df = pd.read_csv(os.path.join(root_path, annotation_file))

    X = df[["img", "fabric"]]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=12345, shuffle=True, stratify=y)

    for data_type, (X, y) in zip(data_types, [(X_train, y_train), (X_test, y_test)]):
        df = X.join(y)
        df.sort_values(by=["class"], ignore_index=True, inplace=True)
        df.to_csv(path_or_buf=os.path.join(root_path, f"{data_type}_annotation.csv"), index=False)
    return None


def main() -> None:
    dataset_root_path = "/home/trong/Downloads/crop"

    generate_annotation(dataset_root_path)
    split_train_test(dataset_root_path, annotation_file="annotation.csv", data_types=["train", "test"], train_size=.8)
    split_train_test(dataset_root_path, annotation_file="train_annotation.csv", data_types=["train", "val"], train_size=.8)
    return None


if __name__ == '__main__':
    main()
