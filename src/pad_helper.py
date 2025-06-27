#!/usr/bin/python
import requests
import base64


# api-endpoint location
URL = "https://pad.crc.nd.edu/index.php?option=com_jbackend&view=request&module=querytojson&resource=list&action=get"


# fetch data, you need a valid API key
# If succesful this will return the DB data in 'data',
# the column headers in 'headers',
# the number of rows in 'items',
# the number of fields in 'fields'
def query_pad_database(project_name, api_key):
    # setup the parameters
    # projects 'queryname':'projects'
    # project samples 'queryname':'projectsamples', 'project_name':'FHI360',
    PARAMS = {
        "queryname": "database",
        "project_name": project_name,
        "api_key": api_key,
    }

    # sending get request and saving the response as response object
    try:
        r = requests.get(url=URL, params=PARAMS)

        # extracting data in json format
        jdata = r.json()

    except Exception as e:
        print("API query error", e)
        return None

    # return
    return jdata

    # helper function to download from pad server


import urllib.request
import os


# actual function
def pad_download(url, file_name=None):

    # use try/catch to fail gracefully
    try:
        # extract filename
        if not file_name:
            file_name = os.path.basename(url)

        # download
        result = urllib.request.urlretrieve(url, file_name)
        return True
    except Exception as e:
        # flag if failed
        print("Could not download", file_name, "Error", e)
        return False
