euro_graph = []

with open(exchange_rates_file, 'r' ) as file:
    csv_reader = csv.DictReader(file)
    for line in csv_reader:
        euro_doc = {
            'date': line['Date'],
            'rate': line['Euro']
        }

        euro_graph.append(euro_doc)
