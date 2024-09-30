import pandas as pd
import requests
from tqdm import tqdm

import time
import os
from os.path import join, exists
from pathlib import Path
import xarray as xr
from datetime import datetime, timedelta
from zipfile import ZipFile, BadZipFile, is_zipfile

class DownloadKit:
    api_url_name = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=contains(Name,'{name}')"
    token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    download_url = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

    def query_by_name(name):
        response_result = None
        json = requests.get(DownloadKit.api_url_name.format(name=name)).json()
        if json['value']:
            print(f"Found ({len(json['value'])}):", name)
            response_result = json['value']
        else:
            print("Could not find: ", name)
        return response_result
    
    def query_by_pos_date(center_wtk, start_date, start_end_date, name=None):
        json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-1' and\
                        OData.CSC.Intersects(area=geography'SRID=4326;{center_wtk}%27) and\
                        contains(Name,'OCN') and\
                        ContentDate/Start gt {start_date}Z and\
                        ContentDate/Start lt {start_end_date}").json()
        response_result = None
        if json['value']:
            print(f"Found ({len(json['value'])}):", name)
            response_result = json['value']
        else:
            print("Could not find: ", name)
        return response_result
        

    
    def is_downloaded(sar_name, folder):
        sar_path = Path(folder) / (sar_name + ".SAFE.zip")
        if sar_path.exists():
            return True
        return False
    
    def return_headers(email, password):
        data = {
            'grant_type': 'password',
            'username': email,
            'password': password,
            'client_id': 'cdse-public'
        }
        token_response = requests.post(
            DownloadKit.token_url, 
            headers=DownloadKit.headers, 
            data=data
        ).json()
        download_headers = {"Authorization": f"Bearer {token_response['access_token']}"}
        return download_headers
    
    def center_wtk(nc_path):
        test_xr = xr.open_dataset(nc_path)
        patch = test_xr[['lat', 'lon']].isel(
            y=test_xr.dims['y'] // 2,
            x=test_xr.dims['x'] // 2,
        )
        return f'POINT({patch.lon.values} {patch.lat.values})'

    def get_start_range(nc_path, hours=6):
        test_xr = xr.open_dataset(nc_path)
        start_date = test_xr.start_date
        input_format = "%d-%b-%Y %H:%M:%S.%f"
        output_format = "%Y-%m-%dT%H:%M:%S"

        delta_time =  timedelta(hours=hours)
        start_date = datetime.strptime(start_date, input_format) - delta_time

        start_end_date = start_date + 2 * delta_time

        start_date = start_date.strftime(output_format)
        start_end_date = start_end_date.strftime(output_format)

        return start_date, start_end_date
    
    def download(email, password, product_id, folder, name=None, overwrite=False):
        folder = Path(folder)
        if name == None:
            name = product_id
        session = requests.Session()
        headers = DownloadKit.return_headers(email, password)
        session.headers.update(headers)
        response = session.get(DownloadKit.download_url.format(product_id=product_id), headers=headers, stream=True)
        total_size = int(response.headers.get('Content-Length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading: {name}', leave=True)
        filepath = folder / (name + '.zip')
        tmp_filepath = (filepath).with_suffix('.tmp') 
        with open(tmp_filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        tmp_filepath.rename(filepath)
        progress_bar.close()
        return filepath

    def onc_zip_to_nc():
        pass

    def unzip(zip_path, unzip_folder):
        zip_path = Path(zip_path)
        print("Unziping:", zip_path.stem)
        with ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                try:
                    zip_ref.extract(member, unzip_folder)
                except BadZipFile:
                    print(f"Imposible to unzip: {zip_path.stem}. file is incomplete or corrupted!")



class DownloadSar:
    def __init__(self, email, password):
        self.password = password
        self.email = email
        self.raw_folder = None

    def download_sls(self, list_of_names, folder, overwrite=False):
        list_of_query = []
        for sar_name in list_of_names:
            if not DownloadKit.is_downloaded(sar_name, folder) or overwrite:
                query = DownloadKit.query_by_name(sar_name)
                if len(query) == 1:
                    query = query[0]
                    DownloadKit.download(self.email, self.password, product_id=query['Id'], folder=folder, name=query['Name'])
                else:
                    print('Could not download because name not found or 2 different data on query to name:', sar_name)

    def download_ocn(self, nc_folder, folder, subproduct_folder, overwrite=False):
        count = 0
        folder = Path(folder)
        subproduct_folder = Path(subproduct_folder)
        nc_folder = Path(nc_folder)
        for nc_file in sorted(nc_folder.glob('*.nc')):
            if not (folder / nc_file.with_suffix('.nc').name).exists() or overwrite:
                center_wtk = DownloadKit.center_wtk(nc_file)
                start_range = DownloadKit.get_start_range(nc_file)
                query = DownloadKit.query_by_pos_date(center_wtk, *start_range, name=nc_file.stem)
                if query:
                    donwload_path = DownloadKit.download(self.email, self.password, query[0]['Id'], folder, name=nc_file.stem)
                    unzip_folder = subproduct_folder / donwload_path.stem
                    DownloadKit.unzip(donwload_path, unzip_folder)
                    measurement_file = list(unzip_folder.glob('*.SAFE'))[0] / 'measurement'
                    ocn_nc_file = list(measurement_file.glob('*.nc'))[0]
                    output_path = donwload_path.with_suffix('.nc')
                    ocn_nc_file.rename(output_path)
                    donwload_path.unlink()
                    count += 1

            





if __name__ == '__main__':
    SAR_TO_NC_GRAPH = "/mnt/camobi_2/PHMG/Sentinel_Acquisition/graphs/ZIP_to_NC.xml"
    FILE_NAME_COLUMN = "NOME DO ARQUIVO"
    NETCDF_FOLDER = "/mnt/camobi_3/new_data/nc_data"
    PATH_TO_GPT = "/home/camobi/snap/bin/gpt"
    PRODUCT_FOLDER = "/mnt/camobi_2/PHMG/Sentinel_Acquisition/raw_folder"
    UNZIP_FOLDER = "/mnt/camobi_2/PHMG/Sentinel_Acquisition/unzip_folder"
    LABEL_PATH = "/mnt/camobi_3/new_data/labels_json"
    IMAGE_PATH = "/mnt/camobi_3/new_data/png_netcdf"
    NO_LAND_XML = "/mnt/camobi_2/PHMG/Sentinel_Acquisition/graphs/NO_LAND.xml"

    raw_nc_name = 'S1A_IW_SLC__1SDV_20141004T154823_20141004T154851_002682_002FE4_C094.SAFE'
    raw_folder = '/mnt/camobi_2/PHMG/Sentinel_Acquisition/raw_folder'
    path_nc_raw = '/mnt/camobi_2/PHMG/Sentinel_Acquisition/linear_nc/0E78_Orb_Cal_Deb_ML.nc'
    test_path = '/mnt/camobi_2/PHMG/delete_now'
    
    data_pd = pd.read_csv("/mnt/camobi_2/PHMG/Sentinel_Acquisition/New_sar_img.csv", header=0)[FILE_NAME_COLUMN]
    data_pd = list(data_pd.dropna(how="all"))
    test = DownloadSar(email="pedro.meirelles@ufba.br", password="Thermal1234@")
    # test.download_sls(data_pd, test_path)
    test.download_ocn(NETCDF_FOLDER, '/mnt/camobi_2/PHMG/temp_ocn', '/mnt/camobi_2/PHMG/unzip_folder_delete')
    # DownloadKit.unzip('/mnt/camobi_2/PHMG/delete_now/0E78.zip', test_path + '/0E78')



        