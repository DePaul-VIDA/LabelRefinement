import os
import gdown
import zipfile

PATH_download = "./download"
root_folder = "./"

url_hug = 'https://drive.google.com/uc?id=1wIkW30nG3EdYFWDaPbtVpDGWvyXb4GyJ'
url_shake = 'https://drive.google.com/uc?id=1ItHfQjM9LHfmluxYSHavQrFna5tzvrzc'
url_sigs = 'https://drive.google.com/uc?id=10Gm0qnZaV0S66C2wy0h-Hgs380mlHg7Y'

print ('Downloading files...')
#print ('Downloading hug')
#gdown.download(url_hug, './download/hug.zip',quiet=False)
print ('Downloading sigs')
gdown.download(url_sigs, './download/sigs.zip',quiet=False)

#print ('Unzipping hug')
#with zipfile.ZipFile(PATH_download + '/hug.zip', 'r') as ziphandler:
#        ziphandler.extractall(root_folder)
print ('Unzipping sigs')
with zipfile.ZipFile(PATH_download + '/sigs.zip', 'r') as ziphandler:
        ziphandler.extractall(root_folder)

