import psutil
import csv
import time

# Set the duration of the monitoring in seconds
duration = 10 * 60

# Open the output file in write mode
with open('system_usage.csv', 'w', newline='') as f:
    # Create a CSV writer object
    writer = csv.writer(f)
    
    # Write the header row to the CSV file
    writer.writerow(['Time', 'CPU Usage (%)', 'Memory Usage (%)'])
    
    # Monitor the system every second for the specified duration
    for i in range(duration):
        # Get the CPU and memory usage percentage of the system
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        
        # Write the usage data to the CSV file
        writer.writerow([i+1, cpu_usage, mem_usage])
        
        # Wait for 1 second before taking the next measurement
        time.sleep(1)