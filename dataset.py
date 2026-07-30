import os
from math import ceil
from random import sample
from shutil import move


folder="emotions" #replace this with the dataset folder


def get_model_class_names(data_directory: str) -> list:
    os.chdir(data_directory)
    class_names = list(filter(os.path.isdir, os.listdir()))
    os.chdir('..')
    return class_names


def list_class_images(data_directory: str, class_name: str, extension_filters = ('jpg', 'png', 'jpeg', 'webp')) -> dict:
    class_images = {}
    for category in ['train', 'test', 'val']:
        class_images[category] = []
        image_directory = os.path.join(data_directory, category, class_name)
        os.makedirs(image_directory, exist_ok=True)
        if not os.listdir(image_directory) and category=="train":
            os.rmdir(image_directory)
            move(os.path.join(data_directory, class_name), os.path.join(data_directory, category))
        for file_name in os.listdir(image_directory):
            name, extension = os.path.splitext(file_name)
            extension = extension.lower().lstrip(".")
            if extension in extension_filters:
                class_images[category].append(file_name)
    return class_images




def split_class_images(data_directory: str, class_name: str, test_percent: float = 0.1, val_percent: float = 0.1):
    class_images = list_class_images(data_directory, class_name)
    total_images = sum(map(len, class_images.values()))
    print(f'There are {total_images} images of the class {class_name}.')
    test_image_count = int(ceil(test_percent * total_images))
    val_image_count = int(ceil(val_percent * total_images))
    train_image_count = total_images - test_image_count - val_image_count
    print(f'Image dataset split: Train={train_image_count}, Test={test_image_count}, Val={val_image_count}.')


    category_counts = {
        "test": test_image_count,
        "val": val_image_count
    }


    for category_name, category_count in category_counts.items():
        if len(class_images[category_name]) > category_count:
            move_image_count = len(class_images[category_name]) - category_count
            randomly_selected_images = sample(class_images[category_name], move_image_count)
            destination_folder = os.path.join(data_directory, 'train', class_name)
            source_folder = os.path.join(data_directory, category_name, class_name)
            for file_name in randomly_selected_images:
                destination_file = os.path.join(destination_folder, file_name)
                source_file = os.path.join(source_folder, file_name)
                os.rename(source_file, destination_file)
            class_images[category_name] = list(sorted(set(class_images[category_name]).difference(randomly_selected_images)))
            class_images['train'] = list(sorted(set(class_images['train']).union(randomly_selected_images)))
        elif len(class_images[category_name]) == category_count:
            print(f'No changes are necessary for class {class_name} {category_name}.')


    for category_name, category_count in category_counts.items():
        if len(class_images[category_name]) < category_count:
            move_image_count = category_count - len(class_images[category_name])
            randomly_selected_images = sample(class_images['train'], move_image_count)
            source_folder = os.path.join(data_directory, 'train', class_name)
            destination_folder = os.path.join(data_directory, category_name, class_name)
            for file_name in randomly_selected_images:
                destination_file = os.path.join(destination_folder, file_name)
                source_file = os.path.join(source_folder, file_name)
                os.rename(source_file, destination_file)
            class_images[category_name] = list(sorted(set(class_images[category_name]).union(randomly_selected_images)))
            class_images['train'] = list(sorted(set(class_images['train']).difference(randomly_selected_images)))


classes = get_model_class_names(folder)
print(f"The classes in our dataset at {folder} are: {classes}")
for class_name in classes:
    split_class_images(folder, class_name)
