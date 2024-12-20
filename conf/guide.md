# **Quick Setup Guide for `data_loader_conf.yaml`**

This guide explains how to configure the `data_loader_conf.yaml` file to set up the `SamplesDataLoader` for data processing before training. Follow these steps to quickly configure and get started.

---

## **Step 1: Define Dataset Paths**

1. **Locate your dataset folder** (e.g., `SPADES` directory).
2. Set the following paths in the YAML file:
   - **`dataset_dir`**: Path to the dataset directory.
   - **`output_dir`**: Directory to save processed data.

**Example:**
```yaml
paths:
  dataset_dir: "/Users/jost/Downloads/SPADES"   Update to your dataset location
  output_dir: "./processed_dataset"             Update for output folder
```

## **Step 2: Choose a Transformation**

1. **Select an event transformation method** based on your model’s input requirements. Each method processes event data differently to prepare it for training.

   - **Available Options**:
     - `two_polarity_time_surface`: Generates time surfaces seperately for both polarities.
     - `two_d_histogram`: Creates a 2D histogram representation of events.
     - `lnes`: Event Hands representation.
     - `lnes2`: Faster LNES algorithm.
     - `to_voxel_grid`: Converts event data into a voxel grid structure.
     - `three_c_representation`: Produces a three-channel representation.

2. Update the `transformation.method` field in the YAML file with your desired method.

**Example:**
```yaml
transformation:
  method: "lnes2"  # Example: Use "lnes2" for Logarithmic Neural Event Surfaces
```

## **Step 3: Configure Filtering**

1. **Set the filtering logic** to refine the events based on specific conditions:
   - **`method`**: Specifies the filtering method to use.
     - For this setup, choose `get_distribution`.

2. **Define filter conditions**:
   - Each condition is written as `[value, bool]`:
     - **`value`**: Interval of the x-axis of the image
     - **`bool`**: The interval of interest

3. Update the `filter` section in the YAML file with your desired method and conditions.

**Example:**
```yaml
filter:
  method: "get_distribution"       # Use this filter method
  parameters:
    conditions:                    # Define filtering conditions
      - [280, false]               # Exclude values below 280
      - [1000, true]               # Include values between 280 and 1000
      - [1280, false]              # Exclude values above 1280
```

## **Step 4: Set Dataset Type**

1. **Specify the dataset type** to indicate the structure of your dataset:
   - **`synthetic`**: Use this for datasets with paired event and label files.
   - **`Real`**: Use this for datasets that only contain event files.

2. Update the `dataset_type` field in the YAML file with your desired type.

**Example:**
```yaml
dataset_type: "synthetic"  # Example: Set "synthetic" for paired event-label datasets
```

## **Complete Setup Example**

Here’s a complete example configuration for the `data_loader_conf.yaml` file:

```yaml
paths:
  dataset_dir: "/Users/jost/Downloads/SPADES"
  output_dir: "./generated_dataset"

transformation:
  method: "lnes2"

filter:
  method: "get_distribution"
  parameters:
    conditions:
      - [280, false]
      - [1000, true]
      - [1280, false]

dataset_type: "synthetic"
```

## **Checklist Before Running**

1. **Paths**: Confirm `dataset_dir` and `output_dir` are correct.
2. **Transformation**: Ensure `transformation.method` matches your model’s needs.
3. **Filter Conditions**: Verify `filter.parameters.conditions` align with the dataset.
4. **Dataset Type**: Set `dataset_type` based on your data format (`synthetic` or `Real`).

---

## **Next Steps**

Once the configuration file is updated, your `SamplesDataLoader` will use these settings to process and prepare your dataset for training. Run the data loader script with the configured YAML file to begin preprocessing.

```bash
python data_loader.py