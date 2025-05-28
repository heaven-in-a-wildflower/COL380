import re
import csv

def process_stats(file_path):
    # Store the results for each section
    results = []

    # Open the stats file and process it
    with open(file_path, "r") as file:
        lines = file.readlines()

    id = None
    stats = {}

    for line in lines:
        # Check for a section demarcator
        if line.startswith(" Performance counter stats for "):
            # Save the current section stats
            if id:
                results.append({"id": id, "stats": stats})

            # Extract the new section name and take the 3rd word
            match = re.search(r"'(.+?)'", line)
            if match:
                full_section = match.group(1)
                words = full_section.split()
                id = [words[2], words[1]] if len(words) > 2 else "Unknown"

            stats = {}  # Reset stats for the new section
            continue

        # Parse cache references
        if "cache-references" in line and "cpu_core" in line:
            match = re.search(r'([\d,]+)\s+cpu_core/cache-references/', line)
            if match:
                stats["cache_references"] = int(match.group(1).replace(",", ""))

        # Parse cache misses
        if "cache-misses" in line and "cpu_core" in line:
            match = re.search(r'([\d,]+)\s+cpu_core/cache-misses/', line)
            if match:
                stats["cache_misses"] = int(match.group(1).replace(",", ""))

        # Instructions
        if "instructions" in line and "cpu_core" in line:
            match = re.search(r'([\d,]+)\s+cpu_core/instructions/', line)
            if match:
                stats["instructions"] = int(match.group(1).replace(",", ""))

        # Instructions
        if "cycles" in line and "cpu_core" in line:
            match = re.search(r'([\d,]+)\s+cpu_core/cycles/', line)
            if match:
                stats["cycles"] = int(match.group(1).replace(",", ""))

        # Parse time elapsed
        if "seconds time elapsed" in line:
            match = re.search(r'([\d\.]+) seconds time elapsed', line)
            if match:
                stats["time_elapsed"] = float(match.group(1))

        # Parse user time
        if "seconds user" in line:
            match = re.search(r'([\d\.]+) seconds user', line)
            if match:
                stats["user_time"] = float(match.group(1))

        # Parse sys time
        if "seconds sys" in line:
            match = re.search(r'([\d\.]+) seconds sys', line)
            if match:
                stats["sys_time"] = float(match.group(1))

    # Save the last section's stats
    if id:
        results.append({"id": id, "stats": stats})

    # Recompute values for each section
    for result in results:
        stats = result["stats"]
        if "cache_references" in stats and "cache_misses" in stats:
            stats["cache_hit_rate"] = round(100 - (stats["cache_misses"] / stats["cache_references"] * 100), 4)

    return results


def write_to_csv(results, csv_file):
    # Define the CSV column headers
    headers = ["Type","Size","Cache References", "Cache Misses", "Cache Hit Rate","Instructions","Cycles","IPC","Time Elapsed", "User Time", "Sys Time"]

    # Write results to the CSV file
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header row

        for result in results:
            type_ = result["id"][1]
            size = result["id"][0]
            stats = result["stats"]
            cache_references = stats.get("cache_references", "N/A")
            cache_misses = stats.get("cache_misses", "N/A")
            cache_hit_rate = stats.get("cache_hit_rate", "N/A")
            instructions = stats.get("instructions", "N/A")
            cycles = stats.get("cycles", "N/A")
            ipc = stats.get("instructions", "N/A") / stats.get("cycles", "N/A")
            time_elapsed = stats.get("time_elapsed", "N/A")
            user_time = stats.get("user_time", "N/A")
            sys_time = stats.get("sys_time", "N/A")

            # Write each row
            writer.writerow([type_,size, cache_references, cache_misses, cache_hit_rate,instructions,cycles,ipc, time_elapsed, user_time, sys_time])


# Example Usage
stats_file = "cache_stats.txt"
csv_file = "cache_stats.csv"

sections_stats = process_stats(stats_file)
write_to_csv(sections_stats, csv_file)
