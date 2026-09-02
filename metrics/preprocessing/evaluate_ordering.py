"""Visualize preprocessing stroke ordering strategies."""

import argparse
import random

from utils.paths import DEFAULT_EVALUATIONS_DIR, DEFAULT_FILTERED_SKETCHES_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize stroke ordering strategies.")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of sketches to sample for comparison (default: 20).",
    )
    return parser.parse_args()

def plot_strokes(strokes, ax, title):
    """
    Plots a list of strokes on a given matplotlib Axis.
    Color transitions from red (first) to blue (last).
    """
    import matplotlib.pyplot as plt

    if not strokes:
        ax.set_title(title + " (Empty)")
        return
        
    num_strokes = len(strokes)
    # Get a colormap (rainbow/jet) to represent sequence
    colormap = plt.get_cmap('rainbow')
    
    for i, stroke in enumerate(strokes):
        # Color goes from red (0) to blue/violet (1)
        color = colormap(1.0 - (i / max(1, num_strokes - 1)))
        
        xs = [pt[0] for pt in stroke]
        ys = [pt[1] for pt in stroke]
        
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8)
        
        # Mark point of origin of the stroke slightly larger
        ax.scatter(xs[0], ys[0], color=color, s=12, zorder=5)
        
    ax.set_title(title)
    ax.invert_yaxis() # Image coordinates (y goes down)
    ax.set_aspect('equal')
    ax.axis('off')

def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from pipeline.ordering import (
        order_directional_bias,
        order_greedy_nearest_neighbor,
        order_tsp,
    )
    from pipeline.vectorization import vectorize_image

    sketches_dir = DEFAULT_FILTERED_SKETCHES_DIR
    output_dir = DEFAULT_EVALUATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all sketches
    all_sketches = list(sketches_dir.rglob("*.png"))
    if not all_sketches:
        print("No sketches found in", sketches_dir)
        return
        
    # Sample 20 sketches randomly (seeded for reproducibility)
    random.seed(42)
    sample_size = min(args.samples, len(all_sketches))
    samples = random.sample(all_sketches, sample_size)
    
    print(f"Evaluating {sample_size} sketches...")
    
    for i, img_path in enumerate(tqdm(samples)):
        try:
            # 1. Vectorize
            strokes = vectorize_image(img_path)
            if not strokes:
                continue
                
            # 2. Re-order
            greedy_strokes = order_greedy_nearest_neighbor(strokes)
            directional_strokes = order_directional_bias(strokes)
            tsp_strokes = order_tsp(strokes)

            # print the number of strokes for each ordering and stroke points
                # print(f"Greedy: {len(greedy_strokes)} strokes")
                # print(f"Greedy: {greedy_strokes}")

                # print(f"Directional: {len(directional_strokes)} strokes")
                # print(f"Directional: {directional_strokes}")

                # print(f"TSP: {len(tsp_strokes)} strokes")
                # print(f"TSP: {tsp_strokes}")
            
            # 3. Plot comparison
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Sample {i+1}: {img_path.name}\n(Red = Early Strokes, Blue = Late Strokes)", fontsize=14)
            
            plot_strokes(greedy_strokes, axs[0], "Greedy Nearest-Neighbor")
            plot_strokes(directional_strokes, axs[1], "Directional Bias")
            plot_strokes(tsp_strokes, axs[2], "TSP Approximation")
            
            plt.tight_layout()
            
            out_file = output_dir / f"eval_{i+1:02d}_{img_path.stem}.png"
            plt.savefig(out_file, dpi=150)
            plt.close(fig)
            
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")
            
    print(f"Evaluation complete. Visualizations saved to {output_dir}")

if __name__ == '__main__':
    main()
