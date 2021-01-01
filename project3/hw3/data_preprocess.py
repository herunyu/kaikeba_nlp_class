import json

with open('city.json', 'r') as f:
    cities = json.load(f)

city_list = []

for i, city in enumerate(cities['cities']):
    city_list.append(city['cityZh'])
    cities['cities'][i]['coordinates'] = (city['lat'], city['lon'])

with open('city_processed.json', 'w') as outfile:
    json.dump(cities, outfile)

with open('citi_list.txt', 'w') as f:
    f.writelines('- lookup: city\n')
    f.writelines('  examples: |\n')
    for city in city_list:
        f.writelines('    - ' + city + '\n')