import csv

# List of labels to keep
labels_to_keep = [
    'yoga', 'braiding hair', 'brushing teeth', 'carrying baby',
    'cleaning floor', 'cleaning gutters', 'cleaning pool',
    'cleaning shoes', 'cleaning toilet', 'cleaning windows',
    'cooking chicken', 'cooking egg', 'cooking sausages',
    'making a cake', 'making a sandwich', 'making bed',
    'making jewelry', 'making pizza', 'making tea',
    'washing dishes', 'washing hands/hair', 'watering plants'
]

# Input and output file paths
input_file = './data/k400/annotations/train.csv'  # Change this to your actual input file path
output_file = './data/k400/annotations/filtered_activities.csv'

# Filter the CSV file
with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write the header row
    header = next(reader)
    writer.writerow(header)
    
    # Write only rows with labels in our list
    for row in reader:
        if row[0] in labels_to_keep:
            writer.writerow(row)

print(f"Filtered data has been written to {output_file}")