#!/usr/bin/env python3
"""
Demonstrate the new hybrid dataset management features.

This example shows how to:
1. List all available datasets (combining dynamic catalog and static mappings)
2. Get comprehensive dataset information
3. Switch between dynamic and static modes
4. Access dataset metadata from the catalog
"""

import pad_analytics as pad


def print_separator():
    print("-" * 60)


def main():
    print("PAD Analytics - Dataset Features Demo")
    print_separator()
    
    # 1. List all available datasets
    print("\n1. Getting dataset list (dynamic mode):")
    datasets_df = pad.get_dataset_list(use_dynamic=True)
    print(f"Found {len(datasets_df)} datasets")
    print("\nDataset names:")
    for idx, row in datasets_df.iterrows():
        source = row.get('Source', 'unknown')
        record_count = row.get('Record Count', 'N/A')
        print(f"  - {row['Dataset Name']} (source: {source}, records: {record_count})")
    
    print_separator()
    
    # 2. Get detailed information about a specific dataset
    print("\n2. Getting detailed dataset information:")
    dataset_name = "FHI2020_Stratified_Sampling"
    info = pad.get_dataset_info(dataset_name)
    
    print(f"\nDataset: {info['name']}")
    print(f"Source: {info.get('source', 'unknown')}")
    
    if 'description' in info:
        print(f"Description: {info['description']}")
    
    if 'record_count' in info:
        print(f"Total Records: {info['record_count']}")
    
    if 'date_published' in info:
        print(f"Published: {info['date_published']}")
    
    if 'models' in info and info['models']:
        print(f"\nModels using this dataset:")
        for model in info['models']:
            print(f"  - Model {model['model_id']}: {model['model_name']}")
    
    print_separator()
    
    # 3. Compare dynamic vs static mode
    print("\n3. Comparing dynamic vs static mode:")
    
    # Dynamic mode (with catalog)
    print("\nDynamic mode datasets:")
    dynamic_datasets = pad.get_dataset_list(use_dynamic=True)
    print(f"  Total: {len(dynamic_datasets)}")
    
    # Static mode (CSV only)
    print("\nStatic mode datasets:")
    static_datasets = pad.get_dataset_list(use_dynamic=False)
    print(f"  Total: {len(static_datasets)}")
    
    # Find datasets only in dynamic catalog
    dynamic_names = set(dynamic_datasets['Dataset Name'])
    static_names = set(static_datasets['Dataset Name'])
    catalog_only = dynamic_names - static_names
    
    if catalog_only:
        print(f"\nDatasets only in dynamic catalog: {catalog_only}")
    
    print_separator()
    
    # 4. Load a dataset using the new system
    print("\n4. Loading dataset with model information:")
    
    # Get dataset for a specific model
    model_id = 16
    print(f"\nLoading dataset for model {model_id}...")
    
    dataset_df = pad.get_dataset_from_model_id(model_id, use_dynamic=True)
    if dataset_df is not None:
        print(f"Dataset shape: {dataset_df.shape}")
        print(f"Training samples: {len(dataset_df[dataset_df['is_train'] == 1])}")
        print(f"Test samples: {len(dataset_df[dataset_df['is_train'] == 0])}")
    
    print_separator()
    
    # 5. Show dataset distribution information
    print("\n5. Dataset file distribution:")
    
    info = pad.get_dataset_info("FHI2020_Stratified_Sampling")
    if 'distribution' in info and info['distribution']:
        print(f"\nFiles in {dataset_name}:")
        for dist in info['distribution'][:5]:  # Show first 5 files
            name = dist.get('name', 'Unknown')
            size = dist.get('contentSize', 0)
            size_mb = size / (1024 * 1024) if size else 0
            print(f"  - {name} ({size_mb:.2f} MB)")
        
        if len(info['distribution']) > 5:
            print(f"  ... and {len(info['distribution']) - 5} more files")
    
    print_separator()
    print("\nDemo completed!")


if __name__ == "__main__":
    main()