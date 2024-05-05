import zipfile
import os
import shutil

def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    unzip_file('data/PigDetect/PigDetect.zip', 'data/PigDetect/')
    dev_images_dir = 'data/PigDetect/dev/images'
    test_images_dir = 'data/PigDetect/test/images'
    images_dir = 'data/PigDetect/images'
    os.makedirs(images_dir, exist_ok=True)

    for image in os.listdir(dev_images_dir):
        shutil.move(os.path.join(dev_images_dir, image), os.path.join(images_dir, image))
    for image in os.listdir(test_images_dir):
        shutil.move(os.path.join(test_images_dir, image), os.path.join(images_dir, image))
    shutil.rmtree('data/PigDetect/dev')
    shutil.rmtree('data/PigDetect/test')
    os.remove('data/PigDetect/PigDetect.zip')


