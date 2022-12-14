import requests

BASE = 'http://127.0.0.1:5000/'


input1 = {
    'ModelID' : [1937],
    'datasource' : [100],
    'YearMade' : [2000],
    'saledate' : ['5/19/2011 0:00']
}

input2 = {
    'ModelID' : [3883],
    'datasource' : [100],
    'YearMade' : [2000],
    'saledate' : ['5/20/2011 0:00'],
    'Enclosure': []
}

input3 = {
    'ModelID': [],          #int
    'datasource' : [],      #int
    'YearMade' : [],        #int
    'ProductGroup' : [],    #str
    'saledate' : [],        #str
    'fiBaseModel' : [],     #str
    'fiModelDesc' : [],     #str
    'Enclosure' : [],       #str
    'Hydraulics' : [],      #str
    'auctioneerID' : []     #str
}

response1 = requests.get(BASE, input1)
response2 = requests.get(BASE, input2)
response3 = requests.get(BASE, input3)
print(response1.json(), response2.json(), response3.json())