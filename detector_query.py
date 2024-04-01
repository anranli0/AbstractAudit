import requests
import json


def predict_hf(s):
    ''' Detection query for Arsh Kashyap AI content detector. '''
    
    url = "https://ai-content-detector2.p.rapidapi.com/analyzePatternsAndPerplexities"
    payload = { "text": s}

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "",
        "X-RapidAPI-Host": "ai-content-detector2.p.rapidapi.com"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.text

def parse_hf(df, col):
    ''' Parser for Arsh Kashyap AI content detector. '''
    
    df['hf_AI'] = df[col].apply(lambda x: json.loads(x)['patternAnalysis']['data'][0]['AI'] if x else None)
    df['hf_human'] = df[col].apply(lambda x: json.loads(x)['patternAnalysis']['data'][0]['Human'] if x else None)
    df['hf_label'] = df[col].apply(lambda x: json.loads(x)['patternAnalysis']['data'][0]['Label'] if x else None)
    return df


def predict_gptzero(s):
    ''' Detection query for GPTZero AI content detector. '''
    
    url = "https://api.gptzero.me/v2/predict/text"
    payload = {
        "document": s,
        "version": ""
    }
    headers = {
        "x-api-key": "",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.text


def parse_gptzero(df, col):
    ''' Parser for GPTZero AI content detector. '''
    
    df['gptzero_average_generated_prob'] = df[col].apply(lambda x: json.loads(x)['documents'][0]['average_generated_prob'] if x else None)
    df['gptzero_completely_generated_prob'] = df[col].apply(lambda x: json.loads(x)['documents'][0]['completely_generated_prob'] if x else None)
    df['gptzero_overall_burstiness'] = df[col].apply(lambda x: json.loads(x)['documents'][0]['overall_burstiness'] if x else None)
    df['gptzero_result_message'] = df[col].apply(lambda x: json.loads(x)['documents'][0]['result_message'] if x else None)
    df['gptzero_classification'] = df[col].apply(lambda x: json.loads(x)['documents'][0]['document_classification'] if x else None)
    return df

def predict_ogai(s):
    ''' Detection query for Originality.ai AI content detector. '''
    
    url = "https://api.originality.ai/api/v1/scan/ai"
    headers = {
        'X-OAI-API-KEY': '',
        'Accept': 'application/json',
    }
    payload = {
        "content": s,
        "aiModelVersion": "2",
        "storeScan": "\"false\""
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.text


def parse_ogai(df, col):
    ''' Parser for Originality.ai AI content detector. '''
    
    df['ogai_score_human'] = df[col].apply(lambda x: json.loads(x)['score']['original'] if x else None)
    df['ogai_score_ai'] = df[col].apply(lambda x: json.loads(x)['score']['ai'] if x else None)
    df['ogai_link'] = df[col].apply(lambda x: json.loads(x)['public_link'] if x else None)
    return df


def predict_sapling(s):
    ''' Detection query for Sapling AI content detector. '''
    
    if len(s) > 8000:
        return
    
    response = requests.post(
        "https://api.sapling.ai/api/v1/aidetect",
        json={
            "key": "",
            "text": s,
        }
    )
    
    if response.status_code == 200:
        return response.text


def parse_sapling(df, col):
    ''' Parser for Sapling AI content detector. '''
    
    df['sapling_score'] = df[col].apply(lambda x: json.loads(x)['score'] if x and isinstance(x, str) else None)
    return df
     
    
    