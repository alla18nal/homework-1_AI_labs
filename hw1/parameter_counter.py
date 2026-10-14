import pandas as pd
import math

# Network definition
data = [
    ["C1", "convolution, then ReLU", 64, "7x7", 2, 3],
    ["P1", "max-pool",               None, "3x3", 2, 0],
    ["C2", "convolution, then ReLU", 192, "5x5", 1, 2],
    ["P2", "max-pool",               None, "3x3", 2, 0],
    ["C3", "convolution, then ReLU", 384, "3x3", 1, 1],
    ["C4", "convolution, then ReLU", 256, "3x3", 1, 1],
    ["C5", "convolution, then ReLU", 256, "3x3", 1, 1],
    ["P3", "max-pool",               None, "3x3", 2, 0],
    ["F6", "dense, then ReLU",       512, None, None, None],
    ["F7", "dense, then ReLU",       256, None, None, None],
    ["F8", "dense",                  10,  None, None, None],
]

df = pd.DataFrame(
    data,
    columns=["Layer", "Operation", "Outputs", "Kernel", "Stride", "Padding"]
)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def conv_output_size(H, kernel, stride, padding):
    return math.floor((H + 2 * padding - kernel) / stride) + 1


def get_kernel_size(kernel):
    """Convert '7x7' -> 7"""
    if pd.isna(kernel):
        return None
    return int(str(kernel).split("x")[0])


# Starting input
H, W, C = 224, 224, 3

shapes = []
parameter_counts = []

for _, row in df.iterrows():

    operation = row["Operation"].lower()
    outputs = row["Outputs"]

    # -----------------------------
    # Convolution
    # -----------------------------
    if "convolution" in operation:

        k = get_kernel_size(row["Kernel"])
        s = row["Stride"]
        p = row["Padding"]

        # Output shape
        H = conv_output_size(H, k, s, p)
        W = conv_output_size(W, k, s, p)

        C_out = outputs

        # Parameters:
        # k^2 * C_in * C_out + C_out
        params = k**2 * C * C_out + C_out

        C = C_out

    # -----------------------------
    # Max pooling
    # -----------------------------
    elif "max-pool" in operation:

        k = get_kernel_size(row["Kernel"])
        s = row["Stride"]
        p = row["Padding"]

        H = conv_output_size(H, k, s, p)
        W = conv_output_size(W, k, s, p)

        # Pooling has no parameters
        params = 0

    # -----------------------------
    # Dense / Fully connected
    # -----------------------------
    elif "dense" in operation:

        # Number of inputs to the first dense layer
        input_features = H * W * C

        # Parameters:
        # N_in * N_out + N_out
        params = input_features * outputs + outputs

        # Represent dense output as 1x1xC
        H = 1
        W = 1
        C = outputs

    # -----------------------------
    # Unknown operation
    # -----------------------------
    else:
        params = 0

    shapes.append(f"{H}x{W}x{C}")
    parameter_counts.append(params)


# Add calculated columns
df["Shape"] = shapes
df["Parameter Count"] = parameter_counts

df