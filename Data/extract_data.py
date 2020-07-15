import requests
import json
import csv
from datetime import date 

# Calling the response
url = "https://covid-19-india-data-by-zt.p.rapidapi.com/GetIndiaTotalCounts"
headers = {
    'x-rapidapi-host': "covid-19-india-data-by-zt.p.rapidapi.com",
    'x-rapidapi-key': "####"
    }
response = requests.request("GET", url, headers=headers)

#Fetching the json data
json_data = json.loads(response.text)

# open a file for writing raw data 
file_name = str(date.today()) + "-overall.csv"
data_file = open(file_name, 'w') 
  
# create the csv writer object 
csv_writer = csv.writer(data_file) 
  
# Counter variable used for writing  
# headers to the CSV file 
count = 0
for raw in json_data['data']: 
    if count == 0: 
  
        # Writing headers of CSV file
        header = []
        for head in raw:
            header.append(head) 
        csv_writer.writerow(header) 
        count += 1
  
    # Writing data of CSV file 
    val = []
    for head in raw:
        val.append(raw[head])
    csv_writer.writerow(val) 

data_file.close() 
