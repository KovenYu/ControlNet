import json

# Define the number of lines
num_lines = 90000

# Prepare the data
data = []
for i in range(num_lines):
    entry = {
        "source": f"/viscam/projects/nerfluid/alis/lhq/lhq_256/perlin_masked_lhq_256/{i:06d}.png",
        "target": f"/viscam/projects/nerfluid/alis/lhq/lhq_256/lhq_256/{i:07d}.png",
        "prompt": "beautiful high-quality detailed nature photo"
    }
    data.append(entry)

# Generate the JSON file
with open('/mnt/data/generated_data.json', 'w') as f:
    for line in data:
        json.dump(line, f)
        f.write('\n')
