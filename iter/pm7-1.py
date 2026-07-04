import os

# Input and output directory paths
xyz_dir = "xyz_files"
gjf_dir = "gjf_files"

# Create the output directory
os.makedirs(gjf_dir, exist_ok=True)

# New Gaussian input header
header = """%nprocshared=32
%mem=5000MW
%chk=TADF.chk
# opt pm7

"""

# Iterate over all .xyz files in the directory
for xyz_file in os.listdir(xyz_dir):
    if xyz_file.endswith(".xyz"):
        file_path = os.path.join(xyz_dir, xyz_file)

        # Read the original XYZ file content
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Get the atom count and coordinate lines
        atom_count = lines[0].strip()  # The first line is the atom count
        atom_lines = ''.join(lines[2:])  # Keep atom information from the third line onward

        # Create the new Gaussian input content
        new_content = header + f"{atom_count}\n\n"  # Header + atom count + blank line
        new_content += "0 1\n"  # Molecular charge and spin multiplicity

        # Format atom coordinate lines
        for line in atom_lines.splitlines():
            parts = line.split()
            atom = parts[0]  # Atom type
            coords = "   ".join(parts[1:])  # Coordinates
            new_content += f"{atom}  {coords}\n"  # Add each atom line

        # Add two blank lines to avoid format errors
        new_content += "\n\n"

        # Generate the new .gjf file path
        gjf_file_path = os.path.join(gjf_dir, xyz_file.replace(".xyz", ".gjf"))

        # Write the converted .gjf file
        with open(gjf_file_path, 'w') as f:
            f.write(new_content)

        print(f"Generated .gjf file: {gjf_file_path}")

print("All files have been converted to .gjf format and saved.")





