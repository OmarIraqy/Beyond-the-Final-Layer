import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def setup_paths():
    """Configure and return all necessary paths."""
    dataset_root = "./Combined_dataset"
    
    paths = {
        'dataset_root': dataset_root,
        'train_csv': os.path.join(dataset_root, "train_classes.csv"),
        'query_csv': os.path.join(dataset_root, "val_query_classes.csv"),
        'gallery_csv': os.path.join(dataset_root, "val_test_classes.csv"),
        'train_img_dir': os.path.join(dataset_root, "image_train"),
        'val_query_img_dir': os.path.join(dataset_root, "image_val_query"),
        'val_gallery_img_dir': os.path.join(dataset_root, "image_val_test"),
    }
    
    print("Paths configured:")
    for key, value in paths.items():
        if 'csv' in key or 'dir' in key:
            print(f"  {key}: {value}")
    
    return paths


def load_data(paths):
    """Load training, query, and gallery CSVs."""
    print("\nLoading CSVs...")
    train_df = pd.read_csv(paths['train_csv'])
    query_df = pd.read_csv(paths['query_csv'])
    gallery_df = pd.read_csv(paths['gallery_csv'])
    
    print(f"Train shape: {train_df.shape}")
    print(f"Query shape: {query_df.shape}")
    print(f"Gallery shape: {gallery_df.shape}")
    print(f"\nTrain columns: {train_df.columns.tolist()}")
    print(f"Query columns: {query_df.columns.tolist()}")
    print(f"Gallery columns: {gallery_df.columns.tolist()}")
    
    # Get max objectID from training set to avoid ID conflicts
    max_train_id = train_df['objectID'].astype(int).max()
    print(f"\nMax training objectID: {max_train_id}")
    
    # Convert objectIDs to int for consistent handling
    train_df['objectID'] = train_df['objectID'].astype(int)
    query_df['objectID'] = query_df['objectID'].astype(int)
    gallery_df['objectID'] = gallery_df['objectID'].astype(int)
    
    return train_df, query_df, gallery_df, max_train_id


def create_unified_id_mapping(query_df, gallery_df, max_train_id):
    """Create unified ID mapping for both query and gallery data."""
    print("\nCreating unified ID mapping...")
    
    # Get unique objectIDs from query and gallery
    query_unique_ids = sorted(query_df['objectID'].unique())
    gallery_unique_ids = sorted(gallery_df['objectID'].unique())
    
    print(f"Query unique IDs: {len(query_unique_ids)}")
    print(f"Gallery unique IDs: {len(gallery_unique_ids)}")
    
    # Find overlapping IDs between query and gallery
    overlapping_ids = set(query_unique_ids) & set(gallery_unique_ids)
    print(f"Overlapping IDs (same person in both): {len(overlapping_ids)}")
    
    # Create a unified mapping
    all_val_ids = sorted(set(query_unique_ids) | set(gallery_unique_ids))
    print(f"Total unique IDs across query and gallery: {len(all_val_ids)}")
    
    # Create mapping from old IDs to new IDs (starting from max_train_id + 1)
    next_available_id = max_train_id + 1
    unified_id_mapping = {}
    
    for old_id in all_val_ids:
        unified_id_mapping[old_id] = next_available_id
        next_available_id += 1
    
    print(f"\nID Mapping Summary:")
    print(f"  {len(unified_id_mapping)} unique validation IDs mapped to " 
          f"{min(unified_id_mapping.values())}-{max(unified_id_mapping.values())}")
    print(f"  Overlapping IDs will keep same ID after remapping: {len(overlapping_ids)}")
    
    # Apply the same mapping to both dataframes
    query_df['objectID_new'] = query_df['objectID'].map(unified_id_mapping)
    gallery_df['objectID_new'] = gallery_df['objectID'].map(unified_id_mapping)
    
    # Verify overlapping IDs consistency
    if len(overlapping_ids) > 0:
        print(f"\nVerifying overlapping IDs are consistent:")
        sample_id = list(overlapping_ids)[0]
        query_new_id = query_df[query_df['objectID'] == sample_id]['objectID_new'].iloc[0]
        gallery_new_id = gallery_df[gallery_df['objectID'] == sample_id]['objectID_new'].iloc[0]
        print(f"  Sample overlapping ID {sample_id}:")
        print(f"    Query remapped to: {query_new_id}")
        print(f"    Gallery remapped to: {gallery_new_id}")
        is_match = query_new_id == gallery_new_id
        print(f"    Match: {is_match} {'✓' if is_match else '✗'}")
    
    return query_df, gallery_df


def rename_images(query_df, gallery_df):
    """Rename images to avoid conflicts using prefixes."""
    print("\nRenaming images...")
    
    # Rename query images
    query_df['imageName_new'] = [f"val_query_{i:06d}.jpg" 
                                  for i in range(len(query_df))]
    
    # Rename gallery images
    gallery_df['imageName_new'] = [f"val_gallery_{i:06d}.jpg" 
                                    for i in range(len(gallery_df))]
    
    print(f"Image renaming summary:")
    print(f"  Query: {query_df.shape[0]} images renamed with val_query_* prefix")
    print(f"  Gallery: {gallery_df.shape[0]} images renamed with val_gallery_* prefix")
    print(f"\nSample mappings:")
    print(query_df[['imageName', 'imageName_new', 'objectID', 'objectID_new']].head())
    print(gallery_df[['imageName', 'imageName_new', 'objectID', 'objectID_new']].head())
    
    return query_df, gallery_df


def copy_images(query_df, gallery_df, paths):
    """Copy validation images to train directory with new names."""
    print("\nCopying images to train directory...")
    
    train_img_dir = paths['train_img_dir']
    val_query_img_dir = paths['val_query_img_dir']
    val_gallery_img_dir = paths['val_gallery_img_dir']
    
    print("Copying val_query images...")
    for idx, row in tqdm(query_df.iterrows(), total=len(query_df)):
        old_img_path = os.path.join(val_query_img_dir, row['imageName'])
        new_img_path = os.path.join(train_img_dir, row['imageName_new'])
        
        if os.path.exists(old_img_path):
            shutil.copy2(old_img_path, new_img_path)
        else:
            print(f"Warning: {old_img_path} not found")
    
    print("Copying val_gallery images...")
    for idx, row in tqdm(gallery_df.iterrows(), total=len(gallery_df)):
        old_img_path = os.path.join(val_gallery_img_dir, row['imageName'])
        new_img_path = os.path.join(train_img_dir, row['imageName_new'])
        
        if os.path.exists(old_img_path):
            shutil.copy2(old_img_path, new_img_path)
        else:
            print(f"Warning: {old_img_path} not found")


def prepare_dataframes(train_df, query_df, gallery_df):
    """Prepare and combine all dataframes."""
    print("\nPreparing dataframes for combining...")
    
    # Update with new names and IDs
    query_df_prepared = query_df.copy()
    query_df_prepared['imageName'] = query_df_prepared['imageName_new']
    query_df_prepared['objectID'] = query_df_prepared['objectID_new']
    query_df_prepared = query_df_prepared[train_df.columns]
    
    gallery_df_prepared = gallery_df.copy()
    gallery_df_prepared['imageName'] = gallery_df_prepared['imageName_new']
    gallery_df_prepared['objectID'] = gallery_df_prepared['objectID_new']
    gallery_df_prepared = gallery_df_prepared[train_df.columns]
    
    print(f"Prepared dataframes for combining:")
    print(f"  Train: {train_df.shape[0]} samples")
    print(f"  Query prepared: {query_df_prepared.shape[0]} samples")
    print(f"  Gallery prepared: {gallery_df_prepared.shape[0]} samples")
    
    # Combine all data
    combined_df = pd.concat([train_df, query_df_prepared, gallery_df_prepared], 
                            ignore_index=True)
    print(f"\nCombined training set: {combined_df.shape[0]} samples")
    print(f"Unique object IDs: {combined_df['objectID'].nunique()}")
    
    # Verify no duplicate image names
    duplicate_images = combined_df['imageName'].value_counts()
    duplicate_images = duplicate_images[duplicate_images > 1]
    if len(duplicate_images) > 0:
        print(f"\nWarning: Found {len(duplicate_images)} duplicate image names!")
        print(duplicate_images)
    else:
        print(f"\n✓ No duplicate image names found")
    
    # Verify objectID range
    all_ids = sorted(combined_df['objectID'].unique())
    print(f"ObjectID range: {min(all_ids)} - {max(all_ids)}")
    print(f"Total unique IDs: {len(all_ids)}")
    
    return combined_df, query_df_prepared, gallery_df_prepared


def save_combined_data(combined_df, train_df, query_df_prepared, gallery_df_prepared, paths):
    """Save combined data and create backups."""
    print("\nSaving combined data...")
    
    train_csv_path = paths['train_csv']
    
    # Create backup of original train.csv
    backup_path = train_csv_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(train_csv_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")
    
    # Save combined dataframe to original path
    combined_df.to_csv(train_csv_path, index=False)
    print(f"✓ Combined training CSV saved to: {train_csv_path}")
    
    # Save to additional reference file
    output_combined_path = train_csv_path.replace('.csv', '_with_val.csv')
    combined_df.to_csv(output_combined_path, index=False)
    print(f"✓ Also saved to: {output_combined_path}")
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    print(f"Original train samples: {train_df.shape[0]}")
    print(f"Added from val_query: {query_df_prepared.shape[0]}")
    print(f"Added from val_gallery: {gallery_df_prepared.shape[0]}")
    print(f"Total combined samples: {combined_df.shape[0]}")
    data_increase = ((combined_df.shape[0] / train_df.shape[0]) - 1) * 100
    print(f"Data increase: {data_increase:.1f}%")
    print(f"Original unique IDs: {train_df['objectID'].nunique()}")
    print(f"Combined unique IDs: {combined_df['objectID'].nunique()}")
    print(f"="*60)


def verify_combined_dataset(combined_df, train_df, query_df_prepared, gallery_df_prepared, paths):
    """Verify the combined dataset integrity."""
    print("\nVerification of combined dataset:")
    
    train_img_dir = paths['train_img_dir']
    
    print(f"\n1. Checking image files exist in train directory:")
    missing_count = 0
    for idx, row in combined_df.iterrows():
        img_path = os.path.join(train_img_dir, row['imageName'])
        if not os.path.exists(img_path):
            missing_count += 1
            if missing_count <= 5:
                print(f"   Missing: {img_path}")
    
    if missing_count == 0:
        print(f"   ✓ All {len(combined_df)} images found in train directory")
    else:
        print(f"   ⚠ {missing_count} images missing!")
    
    print(f"\n2. Distribution by source:")
    source_list = (['original_train'] * len(train_df) + 
                   ['added_query'] * len(query_df_prepared) + 
                   ['added_gallery'] * len(gallery_df_prepared))
    source_counts = pd.Series(source_list).value_counts()
    print(source_counts)
    
    print(f"\n3. Sample of combined data with new names:")
    print(combined_df[['imageName', 'objectID', 'cameraID']].sample(
        min(10, len(combined_df))))
    
    print(f"\n✓ Dataset combination complete!")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Urban-ReID: Combine Validation Data with Training Set")
    print("=" * 60)
    
    # Setup
    paths = setup_paths()
    
    # Load data
    train_df, query_df, gallery_df, max_train_id = load_data(paths)
    
    # Create ID mapping
    query_df, gallery_df = create_unified_id_mapping(query_df, gallery_df, max_train_id)
    
    # Rename images
    query_df, gallery_df = rename_images(query_df, gallery_df)
    
    # Copy images
    copy_images(query_df, gallery_df, paths)
    
    # Prepare and combine dataframes
    combined_df, query_df_prepared, gallery_df_prepared = prepare_dataframes(
        train_df, query_df, gallery_df)
    
    # Save combined data
    save_combined_data(combined_df, train_df, query_df_prepared, gallery_df_prepared, paths)
    
    # Verify
    verify_combined_dataset(combined_df, train_df, query_df_prepared, gallery_df_prepared, paths)


if __name__ == "__main__":
    main()
