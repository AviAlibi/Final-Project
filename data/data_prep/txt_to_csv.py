input_path = './data/custom/co2_million_tonnes.txt'
output_path = './data/custom/co2_million_tonnes.csv'

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        line = line.replace('\t', ',')
        outfile.write(line)
