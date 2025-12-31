import pickle
import pdb

# Path to the .pkl file
# pkl_file_path = "/home/nan/Desktop/NRMFOptim/sprintMesh2/out_sequence_smpl_rots.pkl"
# pkl_file_path = "/home/nan/Desktop/warp_xpbd_neo_hook/steoeomis_tool_tissue/tissue_pts.pkl"
pkl_file_path = "/home/nan/Desktop/warp_xpbd_neo_hook/steoeomis_tool_tissue/tool_3d_poses.pkl"


# Load the .pkl file
with open(pkl_file_path, "rb") as file:
    data = pickle.load(file)


pdb.set_trace()

# Print the structure of the .pkl file
def print_structure(data, indent=0):
    """Recursively print the structure of the data."""
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"{key}: {type(value)}")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        print("  " * indent + f"List of {len(data)} items:")
        if len(data) > 0:
            print_structure(data[0], indent + 1)  # Print structure of the first item
    else:
        print("  " * indent + f"{type(data)}")

print("Structure of the .pkl file:")
print_structure(data)
