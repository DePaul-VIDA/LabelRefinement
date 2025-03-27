import os
import gdown
import zipfile

PATH_download = "./download"
root_folder = "./"

url_hug = 'https://drive.google.com/uc?id=1wIkW30nG3EdYFWDaPbtVpDGWvyXb4GyJ'
url_shake = 'https://drive.google.com/uc?id=1ItHfQjM9LHfmluxYSHavQrFna5tzvrzc'

print ('Downloading files...')
print ('Downloading hug')
gdown.download(url_hug, './download/hug.zip',quiet=False)
print ('Downloading shake')
gdown.download(url_shake, './download/shake.zip',quiet=False)

print ('Unzipping hug')
with zipfile.ZipFile(PATH_download + '/hug.zip', 'r') as ziphandler:
        ziphandler.extractall(root_folder)
print ('Unzipping shake')
with zipfile.ZipFile(PATH_download + '/shake.zip', 'r') as ziphandler:
        ziphandler.extractall(root_folder)

