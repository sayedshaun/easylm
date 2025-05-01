import os


class TextLoader:
    def load(self, dir_or_path: str) -> str:
        if os.path.isdir(dir_or_path):
            return self.load_data_from_dir(dir_or_path)
        else:
            return self.load_data(dir_or_path)

    @staticmethod
    def load_data_from_dir(dir_path: str) -> str:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        txt_files = [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if file.endswith(".txt")
        ]
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {dir_path}")
        return ",".join(txt_files)

    @staticmethod
    def load_data(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path