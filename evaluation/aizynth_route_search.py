import os
import re
import json
from aizynthfinder.aizynthfinder import AiZynthFinder
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw


# ============================================================
# 1. Helper functions
# ============================================================

def ask_string(prompt, default=None, required=False):
    """
    Ask the user to input a string.
    """
    while True:
        if default is None:
            text = input(f"{prompt}: ").strip()
        else:
            text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "" and default is not None:
            return default

        if text == "" and required:
            print("This value is required. Please enter a valid value.")
            continue

        return text


def ask_int(prompt, default, min_value=1):
    """
    Ask the user to input an integer.
    """
    while True:
        text = input(f"{prompt} [default: {default}]: ").strip()

        if text == "":
            return default

        try:
            value = int(text)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue

        if value < min_value:
            print(f"The value must be >= {min_value}.")
            continue

        return value


def sanitize_name(name):
    """
    Convert a molecule name into a safe folder name.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = name.strip("_")

    if name == "":
        name = "molecule"

    return name


def validate_smiles(smiles):
    """
    Check whether the input SMILES can be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES. RDKit failed to parse the input molecule.")

    return smiles


def resolve_config_file():
    """
    Resolve the AiZynthFinder configuration file.

    Priority:
    1. User input at runtime.
    2. FIREFLY_AIZYNTH_CONFIG environment variable.
    3. Firefly-Geni/aizynthfinder_public_data/config.yml.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    public_data_dir = os.path.join(project_root, "aizynthfinder_public_data")

    env_config = os.environ.get("FIREFLY_AIZYNTH_CONFIG", "").strip()

    if env_config:
        default_config = os.path.abspath(os.path.expanduser(env_config))
    else:
        default_config = os.path.join(public_data_dir, "config.yml")

    config_input = ask_string(
        "Enter AiZynthFinder config file",
        default=default_config,
        required=False,
    )

    config_file = os.path.abspath(os.path.expanduser(config_input))

    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            "AiZynthFinder config file was not found: "
            f"{config_file}\n"
            "Please place config.yml in Firefly-Geni/aizynthfinder_public_data/ "
            "or provide an absolute path. The config file should define the stock, "
            "expansion policy, and filter policy files required by AiZynthFinder."
        )

    return config_file


# ============================================================
# 2. Read user inputs
# ============================================================

print("\nAiZynthFinder retrosynthesis route search")
print("=" * 80)

molecule_name = ask_string(
    "Enter molecule name",
    default="molecule",
    required=False
)
molecule_name = sanitize_name(molecule_name)

target_smiles = ask_string(
    "Enter target SMILES",
    required=True
)
target_smiles = validate_smiles(target_smiles)

max_transforms = ask_int(
    "Enter max_transforms",
    default=8,
    min_value=1
)

iteration_limit = ask_int(
    "Enter iteration_limit",
    default=500,
    min_value=1
)

time_limit = ask_int(
    "Enter time_limit in seconds",
    default=300,
    min_value=1
)

min_routes = ask_int(
    "Enter min_routes",
    default=5,
    min_value=1
)

max_routes = ask_int(
    "Enter max_routes",
    default=20,
    min_value=1
)

if max_routes < min_routes:
    print("Warning: max_routes is smaller than min_routes. max_routes will be set equal to min_routes.")
    max_routes = min_routes

config_file = resolve_config_file()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(script_dir, "aizynth_results")
outdir = os.path.join(output_root, molecule_name)
os.makedirs(outdir, exist_ok=True)

print("\nSearch configuration")
print("=" * 80)
print(f"Molecule name    : {molecule_name}")
print(f"Target SMILES    : {target_smiles}")
print(f"Config file      : {config_file}")
print(f"Output directory : {outdir}")
print(f"max_transforms   : {max_transforms}")
print(f"iteration_limit  : {iteration_limit}")
print(f"time_limit       : {time_limit}")
print(f"min_routes       : {min_routes}")
print(f"max_routes       : {max_routes}")
print("=" * 80 + "\n")


# ============================================================
# 3. Initialize AiZynthFinder
# ============================================================

finder = AiZynthFinder(configfile=config_file)

finder.stock.select("zinc")
finder.expansion_policy.select("uspto")
finder.filter_policy.select("uspto")

finder.config.search.max_transforms = max_transforms
finder.config.search.iteration_limit = iteration_limit
finder.config.search.time_limit = time_limit
finder.config.search.return_first = False

finder.config.post_processing.min_routes = min_routes
finder.config.post_processing.max_routes = max_routes

finder.target_smiles = target_smiles


# ============================================================
# 4. Search and build routes
# ============================================================

finder.tree_search(show_progress=True)
finder.build_routes()

statistics = finder.extract_statistics()
print("\nSearch statistics")
print("=" * 80)
print(statistics)
print("=" * 80 + "\n")

statistics_path = os.path.join(outdir, "statistics.json")
with open(statistics_path, "w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)

input_info_path = os.path.join(outdir, "input_parameters.json")
with open(input_info_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "molecule_name": molecule_name,
            "target_smiles": target_smiles,
            "config_file": config_file,
            "stock": "zinc",
            "expansion_policy": "uspto",
            "filter_policy": "uspto",
            "max_transforms": max_transforms,
            "iteration_limit": iteration_limit,
            "time_limit": time_limit,
            "min_routes": min_routes,
            "max_routes": max_routes,
            "output_directory": outdir,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 5. Save AiZynthFinder default route images
# ============================================================

for i, img in enumerate(finder.routes.images[:max_routes]):
    if img is not None:
        img.save(os.path.join(outdir, f"route_{i:03d}.png"))

print(f"Default route images saved to: {outdir}")


# ============================================================
# 6. Save route JSON files
# ============================================================

for i, route_json in enumerate(finder.routes.jsons[:max_routes]):
    route_json_path = os.path.join(outdir, f"route_{i:03d}.json")

    with open(route_json_path, "w", encoding="utf-8") as f:
        if isinstance(route_json, str):
            f.write(route_json)
        else:
            json.dump(route_json, f, indent=2, ensure_ascii=False)

print(f"Route JSON files saved to: {outdir}")


# ============================================================
# 7. Redraw large route images
# ============================================================

MOL_W = 520
MOL_H = 260
X_GAP = 130
Y_GAP = 80
MARGIN = 60

LINE_WIDTH = 4
ARROW_SIZE = 12
ROUNDED_RADIUS = 18

USE_STOCK_BORDER_COLOR = True


def get_mol_children(mol_node):
    """
    AiZynthFinder route JSON structure:
    molecule node -> reaction node -> molecule children.
    """
    rxn_children = mol_node.get("children", [])

    if not rxn_children:
        return []

    rxn_node = rxn_children[0]
    return rxn_node.get("children", [])


def mol_to_panel(smiles, in_stock=False):
    """
    Draw one molecule panel without text labels.
    """
    panel = Image.new("RGB", (MOL_W, MOL_H), "white")
    draw_obj = ImageDraw.Draw(panel)

    if USE_STOCK_BORDER_COLOR:
        border_color = (80, 170, 80) if in_stock else (230, 160, 60)
    else:
        border_color = (120, 120, 120)

    draw_obj.rounded_rectangle(
        [3, 3, MOL_W - 4, MOL_H - 4],
        radius=ROUNDED_RADIUS,
        outline=border_color,
        width=LINE_WIDTH,
    )

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        mol_img = Draw.MolToImage(
            mol,
            size=(MOL_W - 30, MOL_H - 30),
        ).convert("RGB")

        panel.paste(mol_img, (15, 15))

    return panel


def collect_layout(node):
    """
    First calculate layout as target-left and precursors-right,
    then flip it into precursors-left and target-right.
    """
    positions = {}
    leaf_counter = [0]

    def _layout(n, depth):
        children = get_mol_children(n)

        if not children:
            y = leaf_counter[0] * (MOL_H + Y_GAP)
            leaf_counter[0] += 1
        else:
            child_ys = []
            for child in children:
                _layout(child, depth + 1)
                child_ys.append(positions[id(child)][1])
            y = sum(child_ys) / len(child_ys)

        x = depth * (MOL_W + X_GAP)
        positions[id(n)] = (x, y)

    _layout(node, 0)

    max_x = max(x for x, y in positions.values())
    max_y = max(y for x, y in positions.values())

    flipped_positions = {}

    for node_id, (x, y) in positions.items():
        flipped_x = max_x - x
        flipped_positions[node_id] = (flipped_x, y)

    canvas_w = int(max_x + MOL_W + 2 * MARGIN)
    canvas_h = int(max_y + MOL_H + 2 * MARGIN)

    return flipped_positions, canvas_w, canvas_h


def draw_route(route_dict, outfile):
    """
    Redraw a retrosynthesis route as a large image.
    """
    positions, canvas_w, canvas_h = collect_layout(route_dict)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw_obj = ImageDraw.Draw(canvas)

    def _draw_edges(n):
        children = get_mol_children(n)
        x_parent, y_parent = positions[id(n)]

        for child in children:
            x_child, y_child = positions[id(child)]

            start = (
                x_child + MOL_W + MARGIN,
                y_child + MOL_H / 2 + MARGIN,
            )
            end = (
                x_parent + MARGIN,
                y_parent + MOL_H / 2 + MARGIN,
            )
            mid_x = (start[0] + end[0]) / 2

            draw_obj.line(
                [start, (mid_x, start[1]), (mid_x, end[1]), end],
                fill="black",
                width=LINE_WIDTH,
            )

            draw_obj.polygon(
                [
                    (end[0], end[1]),
                    (end[0] - ARROW_SIZE, end[1] - ARROW_SIZE / 2),
                    (end[0] - ARROW_SIZE, end[1] + ARROW_SIZE / 2),
                ],
                fill="black",
            )

            _draw_edges(child)

    def _draw_nodes(n):
        x, y = positions[id(n)]
        smiles = n.get("smiles", "")
        in_stock = n.get("in_stock", False)

        panel = mol_to_panel(smiles, in_stock=in_stock)
        canvas.paste(panel, (int(x + MARGIN), int(y + MARGIN)))

        for child in get_mol_children(n):
            _draw_nodes(child)

    _draw_edges(route_dict)
    _draw_nodes(route_dict)

    canvas.save(outfile, dpi=(600, 600))


for i, route_dict in enumerate(finder.routes.dicts[:max_routes]):
    outfile = os.path.join(outdir, f"route_{i:03d}_redraw.png")
    draw_route(route_dict, outfile)

print(f"Redrawn large route images saved to: {outdir}")


# ============================================================
# 8. Final summary
# ============================================================

print("\nAiZynthFinder route search completed.")
print(f"All results have been saved to: {outdir}")
print("Generated files include:")
print("  - route_XXX.png")
print("  - route_XXX.json")
print("  - route_XXX_redraw.png")
print("  - statistics.json")
print("  - input_parameters.json")





