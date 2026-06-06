import re
from pathlib import Path
import subprocess

import pandas as pd

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "sqlite"
OBJ_DIR = BASE_DIR / "objs"
DATASETS_DIR = BASE_DIR / "datasets"

GCC_FLAGS = [
    "-c",
    "-g",
    "-O2",
    "-I./coreutils/src",
    "-I./coreutils/lib",
    "-I./coreutils",
]


class Extractor:
    def get_funcs_from_binary(self, filepath: Path):
        functions = {}

        try:
            result = subprocess.run(
                ["objdump", "-d", "-s", str(filepath)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )

            # e.g., "0000000000000000 <my_function>:"
            func_blocks = re.split(
                pattern=r"^([0-9a-fA-F]+)\s+<([^>]+)>:",
                string=result.stdout,
                flags=re.MULTILINE,
            )

            for i in range(1, len(func_blocks), 3):
                func_name = func_blocks[i + 1]
                func_body = func_blocks[i + 2]

                hex_bytes = []

                for line in func_body.split("\n"):
                    match = re.search(
                        r"^\s*[0-9a-f]+:\s+(([0-9a-f]{2}\s+)+)", line
                    )

                    if match:
                        cleared_hex = match.group(1).strip()
                        hex_bytes.append(cleared_hex)

                if hex_bytes:
                    functions[func_name] = " ".join(hex_bytes)

        except Exception as e:
            print(f"Error disassembly {str(filepath)}: {e}")

        return functions

    def get_c_funcs(self, filepath: Path):
        with filepath.open(mode="r") as f:
            content = f.read()

        pattern = (
            r"(?:[a-zA-Z_][a-zA-Z0-9_*]*\s+)+"
            r"([a-zA-Z_][a-zA-Z0-9_]*)"
            r"\s*\([^)]*\)\s*\{([\s\S]*?)\}"
        )

        matches = re.findall(pattern, content)

        return {name: body.strip() for name, body in matches}

    def extract(self, raw_data_dir: Path) -> pd.DataFrame:
        files = raw_data_dir.rglob("*.c")

        rows = []

        for c_file in files:
            obj_file = OBJ_DIR / c_file.name.replace(".c", ".o")

            result = subprocess.run(
                ["gcc"] + GCC_FLAGS + [str(c_file), "-o", str(obj_file)],
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                continue

            binary_funcs = self.get_funcs_from_binary(obj_file)
            c_funcs = self.get_c_funcs(c_file)

            for func_name, func_body in binary_funcs.items():
                if func_name in c_funcs and len(c_funcs[func_name]) > 10:
                    rows.append(
                        [
                            func_name,
                            binary_funcs[func_name],
                            c_funcs[func_name],
                        ]
                    )

        return pd.DataFrame(rows, columns=["name", "hex", "target"])


def main():
    extractor = Extractor()
    df = extractor.extract(RAW_DATA_DIR)

    df.to_parquet(
        DATASETS_DIR / (DATASETS_DIR / "dataset-v1.parquet"),
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print(f"Saved {len(df)} new funcions")


if __name__ == "__main__":
    main()
