import numpy as np
import os
import sqlite3
import shutil

"""
Wie Dataset splitten?

DIES IST NUR EIN EINMAL SCRIPT (siehe Beschreibung)

1. Common Split:
Wir wollen nur auf common trainieren und dann zeigen, dass unsere Variante auch uncommon besser klassifizieren kann.

1.1 Speichere alle Bildpfade von <classname>-<common_setting> in einer Liste ab
1.2 Suche in den Annotations, ob ein Bild dieser Liste eigentlich eine andere Klasse ist und speichre das Bild entsprechend um
--> Bsp. Wenn in "dog-indoors" ein "cat-water" enthalten ist, dann entferne den Pfad einfach aus "dog_common" (cat-water ist uncommon)
--> Bsp. Wenn in "dog-indoors" ein "cat-indoors" enthalten, ist, dann schiebe den Pfad in "cat_common" (cat-indoors ist common)

1.3. Speichere dieses dict als csv Datei ab unter "classes_common" (dann muss das nicht jedes Mal gemacht werden)

1.4 Split vor jedem Lauf:  -> das wird in focus.py gemacht, da es jedes Mal aufgerufen werden muss!!
1.4.1 Lese die csv Datei mit dem dict aus
1.4.2 Führe den random shuffle mit seed aus für die Pfadliste jeder Klasse
1.4.3 Die ersten 8 Bilder gehen in den train_split (je nach 2, 4, 8 epc werden nur die ersten 2, 4, 8 Bilder zum Training benutzt). Für die verbleibenden Bilder führe 30/70 (oder so) Val/Test-Split durch

2. Uncommon Split:
Auf diesem Split wird nur getestet! Führe deshalb die gleiche Prozedur wie in 1 durch bis inkl. 1.3 jedoch nur für uncommon Klassen

3. common + uncommon split:
Führe nochmal das selbe durch, aber diesmal für alle Bilder
"""

# Root of focus as it comes from the download
focus_root = "/data/vilab06/focus"
# focus_root = r"D:\Studium\TUDarmstadt\WiSe23_24\DLCV\datasets\focus\focus"

# Should the correctly sorted images be saved in a new dir?
save_images_to_new_focus_dir = True
# Location of new dir
focus_new_root = "/data/vilab06/focus_our"

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

# Settings in .../focus
settings = ['desert', 'fog', 'forest', 'grass', 'indoors', 'night', 'rain', 'snow', 'street', 'water']

times = {
    "day": 0,
    "night": 1,
    "none": 2,
}

weathers = {
    "cloudy": 0,
    "foggy": 1,
    "partly cloudy": 2,
    "raining": 3,
    "snowing": 4,
    "sunny": 5,
    "none": 6,
}

locations = {
    "forest": 0,
    "grass": 1,
    "indoors": 2,
    "rocks": 3,
    "sand": 4,
    "street": 5,
    "snow": 6,
    "water": 7,
    "none": 8,
}

uncommon = {
    0: {  # truck
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 6, 7},
    },
    1: {  # car
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 6, 7},
    },
    2: {  # plane
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 2, 3, 4, 6, 7},
    },
    3: {  # ship
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 1, 2, 3, 4, 5, 6},
    },
    4: {  # cat
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 3, 4, 6, 7},
    },
    5: {  # dog
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {0, 3, 6},
    },
    6: {  # horse
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 5, 6, 7},
    },
    7: {  # deer
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 3, 4, 5, 6, 7},
    },
    8: {  # frog
        "time": {},
        "weather": {1, 3, 4},
        "locations": {2, 5, 6},
    },
    9: {  # bird
        "time": {1},
        "weather": {1, 3, 4},
        "locations": {2, 5, 6},
    },
}


# This is just a helper class for the annotations.db
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

    @staticmethod
    def stringify(values):
        if len(values) == 1:
            return f"('{values[0]}')"
        else:
            return str(tuple(values))


def is_time_uncommon(category_label, time_label):
    uncommon_settings = uncommon[category_label.item()]
    return time_label.item() in uncommon_settings["time"]


def is_weather_uncommon(category_label, weather_label):
    uncommon_settings = uncommon[category_label.item()]
    return weather_label.item() in uncommon_settings["weather"]


def is_locations_uncommon(category_label, locations_label):
    uncommon_settings = uncommon[category_label.item()]
    return not set(np.nonzero(locations_label.tolist())[0]).isdisjoint(
        uncommon_settings["locations"]
    )


def count_uncommon_attributes(
        category_label, time_label, weather_label, locations_label
):
    return sum(
        [
            is_time_uncommon(category_label, time_label),
            is_weather_uncommon(category_label, weather_label),
            is_locations_uncommon(category_label, locations_label),
        ]
    )


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

    # Save all images of a class to a new folder
    if save_images_to_new_focus_dir:
        os.makedirs(focus_new_root, exist_ok=True)

        for class_name, file_paths in class_images_dict.items():
            class_dir = os.path.join(focus_new_root, class_name)
            os.makedirs(class_dir, exist_ok=True)

            for i, file_path in enumerate(file_paths):
                original_file_path = os.path.join(focus_root, file_path)
                new_file_path = os.path.join(class_dir, f"{i}.jpeg")

                # Copy the file to the new location
                shutil.copy(file_path, new_file_path)
