import json
import http.client
from urllib.parse import urlencode

CLIENT_ID = "aa94502c-1518-4c17-9fba-88f7e6580baf"
API_URL = "api.webstat.banque-france.fr"


def get_dataset_list():
    conn = http.client.HTTPSConnection(API_URL)

    headers = {
        'X-IBM-Client-Id': CLIENT_ID,
        'accept': "application/json"
    }

    conn.request("GET", "/webstat-fr/v1/catalogue?format=json", headers=headers)

    res = conn.getresponse()
    data = res.read()

    annuaire_dict = data.decode("utf-8")
    annuaire_dict = json.loads(annuaire_dict)
    return annuaire_dict

def get_series_by_dataset_name(dataset_name):
    conn = http.client.HTTPSConnection(API_URL)

    conn.request("GET", 
                 f"/webstat-fr/v1/catalogue/{dataset_name}?format=json", 
                 headers={
                    'X-IBM-Client-Id': CLIENT_ID,
                    'accept': "application/json"
                })

    try:
        res = conn.getresponse()
        data = res.read()

        raw_result = data.decode("utf-8")
        if "The document has moved" in raw_result:
            return []
        else:
            datasets_available = json.loads(raw_result)
            return datasets_available
    except Exception as e:
        raise Exception(dataset_name, raw_result, e.args)
    

def get_serie_observations(dataset, serie_key, start_period="2000-01-01", end_period="2019-12-31"):
    querystr_dict = {
        "format": "json",
        "detail": "dataonly",
        "startPeriod" : start_period,
        "endPeriod" : end_period,
        # "lastNObservations": last_observations=1,
        # "firstNObservations": first_observations=1,
    }
    querystr = urlencode(querystr_dict)

    headers = {
        'X-IBM-Client-Id': CLIENT_ID,
        'accept': "application/json"
    }

    conn = http.client.HTTPSConnection(API_URL)
    conn.request("GET", f"/webstat-fr/v1/data/{dataset}/{serie_key}?{querystr}", headers=headers)

    res = conn.getresponse()
    data = res.read()

    serie_observations = json.loads(data.decode("utf-8"))
    if "seriesObs" in serie_observations:
        return serie_observations["seriesObs"][0]
    else:
        return None


def get_dataset_observations(dataset, start_period="2000-01-01", end_period="2019-12-31"):
    querystr_dict = {
        "format": "json",
        "detail": "dataonly",
        "startPeriod" : start_period,
        "endPeriod" : end_period,
        # "lastNObservations": last_observations=1,
        # "firstNObservations": first_observations=1,
    }
    querystr = urlencode(querystr_dict)

    headers = {
        'X-IBM-Client-Id': CLIENT_ID,
        'accept': "application/json"
    }

    conn = http.client.HTTPSConnection(API_URL)
    conn.request("GET", f"/webstat-fr/v1/data/{dataset}?{querystr}", headers=headers)

    res = conn.getresponse()
    data = res.read()

    dataset_observations = json.loads(data.decode("utf-8"))
    return dataset_observations