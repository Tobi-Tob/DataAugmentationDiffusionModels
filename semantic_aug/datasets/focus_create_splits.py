import os
import sqlite3
import shutil
import random

# Root of focus as it comes from the download
focus_root = "/data/vilab06/focus_original"
# focus_root = r"D:\Studium\TUDarmstadt\WiSe23_24\DLCV\datasets\focus\focus"

# Should the correctly sorted images be saved in a new dir?
save_images_to_new_focus_dir = True
# Location of new dir
focus_new_root = "/data/vilab06/focus"

categories = {
        "truck": 0,
        "car": 1,
        "plane": 2,
        "ship": 3,
        "cat": 4,
        "dog": 5,
        "horse": 6,
        "deer": 7,
        "frog": 8,
        "bird": 9,
    }

# Settings in focus dataset
settings = ['desert', 'fog', 'forest', 'grass', 'indoors', 'night', 'rain', 'snow', 'street', 'water']


# This is just a helper class to read the annotations.db
class BGVarDB:

    ANNOTATIONS_TABLE = "annotations"

    def __init__(self, path) -> None:
        self._connection = sqlite3.connect(path)
        self._cursor = self._connection.cursor()

        self._cursor.execute(
            f"CREATE TABLE if not exists {BGVarDB.ANNOTATIONS_TABLE} (file_name TEXT PRIMARY KEY, category TEXT, time TEXT, weather TEXT, locations TEXT, humans TEXT)",
        )

    def read_entries(self, categories=None, times=None, weathers=None, locations=None, humans=None):

        query = f"SELECT file_name, category, time, weather, locations FROM {BGVarDB.ANNOTATIONS_TABLE} WHERE "
        conditions = []
        if categories is not None:
            conditions.append(f"category IN {BGVarDB.stringify(categories)}")
        if times is not None:
            conditions.append(f"time IN {BGVarDB.stringify(times)}")
        if weathers is not None:
            conditions.append(f"weather IN {BGVarDB.stringify(weathers)}")
        if locations is not None:
            conditions.append(f"locations LIKE '%{', '.join((sorted(locations)))}%'")
        if humans is None:
            conditions.append(f"humans = 'no'")

        query += " AND ".join(conditions)
        results = self._cursor.execute(query)
        yield from results


if __name__ == "__main__":
    # Initialize a dictionary to hold the class names as keys and lists of file paths as values
    class_images_dict = {class_name: [] for class_name in categories}

    # Extract all class_paths from the subdirectories
    for class_name in categories:
        for setting in settings:

            folder_name = f"{class_name}-{setting}"
            folder_path = os.path.join(focus_root, folder_name)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.jpeg'):
                        file_path = os.path.join(folder_path, file)
                        # Add the image path to the corresponding class in the dictionary
                        class_images_dict[class_name].append(file_path)

    # Relocate files if they have other annotation
    annotations_db = BGVarDB(r"D:\Studium\TUDarmstadt\WiSe23_24\DLCV\datasets\focus\focus\annotations.db")
    for class_name in categories.keys():
        rows = annotations_db.read_entries(categories=[class_name])
        for img_path, actual_class, *_ in rows:
            if actual_class not in img_path:
                # Extract wrong class label from path
                wrong_label = "none"
                for c in categories.keys():
                    if c in img_path:
                        wrong_label = c
                if wrong_label == "none":
                    raise ValueError(f"Unknown class label in: {img_path}")
                # Move file paths of falsely located image to correct class label
                class_images_dict[wrong_label].remove(img_path)
                class_images_dict[actual_class].appen(img_path)

    # Create train-val and test split for all 10 classes. Save the images to a new directory
    if save_images_to_new_focus_dir:
        os.makedirs(focus_new_root, exist_ok=True)  # new root dir
        train_val_dir = os.path.join(focus_new_root, "train-val")
        test_dir = os.path.join(focus_new_root, "test")
        os.makedirs(train_val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name, file_paths in class_images_dict.items():
            # Shuffle the list of file paths
            random.shuffle(file_paths)

            # Split the list into train-val and test
            train_val_paths = file_paths[:108]
            test_paths = file_paths[108:]

            # Create directories for train-val and test inside each class directory
            class_train_val_dir = os.path.join(train_val_dir, class_name)
            class_test_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_train_val_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            # Copy the first 108 files to the train-val directory
            for i, file_path in enumerate(train_val_paths):
                original_file_path = os.path.join(focus_root, file_path)
                new_file_path = os.path.join(class_train_val_dir, f"{class_name}_{i}.jpeg")
                shutil.copy(file_path, new_file_path)

            # Copy the rest of the files to the test directory
            for i, file_path in enumerate(test_paths):
                original_file_path = os.path.join(focus_root, file_path)
                new_file_path = os.path.join(class_test_dir, f"{class_name}_{len(train_val_paths) + i}.jpeg")
                shutil.copy(file_path, new_file_path)
