import json
import pandas as pd

# Load the JSON data from the uploaded file
file_path = 'all_test_cases_2025-04-30_12-26-33.json'
try:
  with open(file_path, 'r') as f:
    data = json.load(f)
except FileNotFoundError:
  print(f"Error: The file '{file_path}' was not found.")
  data = None
except json.JSONDecodeError:
  print(f"Error: The file '{file_path}' is not valid JSON.")
  data = None

if data:
  # Extract encoding times and query times
  encoding_times = []
  query_times = []

  # Ensure data is a list before proceeding
  if not isinstance(data, list):
    print("Error: JSON data is not in the expected list format.")
  else:
    for entry in data:
      # Check if 'encoding-time' key exists and the value is a number
      if 'encoding-time' in entry and isinstance(entry['encoding-time'], (int, float)):
        encoding_times.append(entry['encoding-time'])
      else:
        # Print a warning or handle missing/invalid encoding time
        print(f"Warning: Missing or invalid 'encoding-time' in entry: {entry.get('filename', 'N/A')}")


      # Check if 'queries' key exists and the value is a list
      if 'queries' in entry and isinstance(entry['queries'], list):
        for query_info in entry['queries']:
          # Check if query_info is a list with at least 3 elements and the third element is a number
          if isinstance(query_info, list) and len(query_info) >= 3 and isinstance(query_info[2], (int, float)):
            query_times.append(query_info[2])
          else:
             # Print a warning or handle invalid query format
             print(f"Warning: Invalid query format or missing query time in entry: {entry.get('filename', 'N/A')}")
      else:
        # Print a warning or handle missing 'queries' list
        print(f"Warning: Missing or invalid 'queries' list in entry: {entry.get('filename', 'N/A')}")


    # Calculate averages
    if encoding_times:
      average_encoding_time = sum(encoding_times) / len(encoding_times)
    else:
      average_encoding_time = 0
      print("Warning: No valid encoding times found to calculate the average.")


    if query_times:
      average_query_time = sum(query_times) / len(query_times)
    else:
      average_query_time = 0
      print("Warning: No valid query times found to calculate the average.")


    # Print the results
    print(f"Total entries processed: {len(data)}")
    print(f"Total encoding times collected: {len(encoding_times)}")
    print(f"Total query times collected: {len(query_times)}")
    print(f"\nAverage Image Encoding Time: {average_encoding_time:.4f} seconds")
    print(f"Average Query Time: {average_query_time:.4f} seconds")