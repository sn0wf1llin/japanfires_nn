__author__ = 'MA573RWARR10R'
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import threading
import os
import urllib
from GoogleScraper import scrape_with_config, GoogleSearchError
import requests


class FetchResource(threading.Thread):
    """Grabs a web resource and stores it in the target directory.
    Args:
        target: A directory where to save the resource.
        urls: A bunch of urls to grab
    """

    def __init__(self, target, urls):
        super().__init__()
        self.target = target
        self.urls = urls

    def run(self):
        for url in self.urls:
            url = urllib.parse.unquote(url)
            with open(os.path.join(self.target, url.split('/')[-1]), 'wb') as f:
                try:
                    content = requests.get(url).content
                    f.write(content)
                except Exception as e:
                    pass
                print('[+] Fetched {}'.format(url))


class RecHelper:
    def __call__(self, key_phrase, pages_count=1, threads_count=10, target_directory="gotten_images"):
        self.image_search(key_phrase=key_phrase, pages_count=pages_count, threads_count=threads_count,
                          target_directory=target_directory)

    @staticmethod
    def quick_clear_from_zerobytes(target_directory):
        prefix = target_directory + "/"

        if os.path.exists(target_directory):
            for f in os.listdir(target_directory):
                old_fname, old_ext = os.path.splitext(prefix + f)

                if old_ext == ".gif":
                    os.remove(prefix + f)

                try:
                    im = Image.open(prefix + f)
                    if im.height > 3*im.width or im.width > 3*im.height:
                        os.remove(prefix + f)

                except Exception as e:
                    try:
                        os.remove(prefix + f)
                    except FileNotFoundError:
                        print("can''t remove %s" % f)
                        pass

    @staticmethod
    def quick_rename_images(target_directory, ren_pref, start=11922):
        prefix = target_directory + "/"

        for index, imf in enumerate(os.listdir(target_directory)):
            if imf != '.DS_Store':
                extension = os.path.splitext(imf)[1]
                if extension != '.jpg':
                    extension = '.jpg'
                new_imf = ren_pref + "%02d%s" % (index+start, extension)
                os.rename(prefix + imf, prefix + new_imf)

    @staticmethod
    def quick_resize_images(target_directory, size=(150, 150)):
        prefix = target_directory + "/"

        for item in os.listdir(target_directory):
            if item != '.DS_Store':
                if os.path.isfile(prefix + item):
                    fname, ext = os.path.splitext(item)

                    try:
                        im = Image.open(prefix + item)

                        if im.mode != 'RGB':
                            im.convert('RGB')

                        im_resized = im.resize(size, Image.ANTIALIAS)
                        new_filename = "data/train/var/" + fname + ext
                        im_resized.save(new_filename)
                    except Exception as e:
                        os.remove(prefix + item)

    @staticmethod
    def image_search(key_phrase, threads_count, pages_count, target_directory, search_engines=None):
        if not search_engines:
            search_engines = ['google', 'baidu', 'yandex', 'bing', 'yahoo']

        config = {
            'keywords': [key_phrase],
            'search_engines': search_engines,
            'search_type': 'image',
            'scrape_method': 'selenium',
            'do_caching': False,
            'num_pages_for_keyword': str(pages_count)
        }

        try:
            search = scrape_with_config(config)
        except GoogleSearchError as e:
            print(e)

        image_urls = []
        print("\t\t\t --- ", len(image_urls))

        for serp in search.serps:
            image_urls.extend(
                [link.link for link in serp.links]
            )

        print('[i] Going to scrape {num} images and saving them in "{dir}"'.format(
            num=len(image_urls),
            dir=target_directory
        ))

        try:
            os.mkdir(target_directory)
        except FileExistsError:
            pass

        # fire up 100 threads to get the images - threads_count

        threads = [FetchResource(target_directory, []) for i in range(threads_count)]

        while image_urls:
            for t in threads:
                try:
                    t.urls.append(image_urls.pop())
                except IndexError as e:
                    break

        threads = [t for t in threads if t.urls]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        return True

    @staticmethod
    def generate_variants_of_data(data_directory, batches_per_image=2):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        prefix = data_directory + '/'
        for im in os.listdir(data_directory):
            if im != '.DS_Store' and os.path.isfile(prefix + im):
                fname, ext = os.path.splitext(im)

                itest = load_img(prefix + im)
                iarray = img_to_array(itest)
                iarray = iarray.reshape((1,) + iarray.shape)

                i = 0
                for batch in datagen.flow(iarray, batch_size=1, save_to_dir=data_directory,
                                          save_prefix=fname, save_format='jpeg'):
                    i += 1
                    if i > batches_per_image:
                        break  # otherwise the generator would loop indefinitely


if __name__ == "__main__":
    im_gr = RecHelper()

    # i = 0
    # while i <= 10:
    #     i += 1
    #     im_gr("candlestick+chart")

    # with open("random_words.txt", 'r') as f:
    #     s = f.readline()
    #
    # for sword in s.split():
    #     print(sword)
    #     if im_gr(sword):
    #         print("GOT")

    # im_gr.quick_clear_from_zerobytes("gotten_images")

    # im_gr.quick_rename_images("gotten_images", ren_pref="trash")

    # im_gr.generate_variants_of_data("test_data", batches_per_image=5)

    # im_gr.quick_resize_images("data/train/trash")
