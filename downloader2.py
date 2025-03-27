import os
import zipfile

import gdown
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd

PATH_download = "./"
print ('make directory download')
#os.makedirs(PATH_download)

id_mmdet = "1dMhHYEdAIqUVDlGnxRyulIFWCvTqkzeJ"
mmdet_name = './mmdet.zip'
id_mmtrack = "1tTq1E6Yh8BP79MVIvWS96JMYmRH3cF0K"
mmtrack_name = './mmtrack.zip'
id_mmpose = "16k4W-9KC3EyGCXC2DZ4rVGP_B79iAjYL"
mmpose_name = './mmpose.zip'

print ('Downloading files...')
gdd.download_file_from_google_drive(file_id=id_mmdet,
                                    dest_path=mmdet_name,
                                    unzip=True)
gdd.download_file_from_google_drive(file_id=id_mmtrack,
                                    dest_path=mmtrack_name,
                                    unzip=True)
gdd.download_file_from_google_drive(file_id=id_mmpose,
                                    dest_path=mmpose_name,
                                    unzip=True)
