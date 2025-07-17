def analyze_feature_correlations(metadata_df, target_column='quantity', num_samples=30):
    """Analyze correlations between RGB features and target variable."""
    
    print(f"🔄 Analyzing correlations for {num_samples} samples...")
    
    # Extract features and targets
    feature_data = []
    targets = []
    
    for i in range(min(num_samples, len(metadata_df))):
        row = metadata_df.iloc[i]
        features, _ = extract_rgb_features(row['url'])
        
        if features and len(features) > 100:
            # Convert to ordered feature vector
            feature_vector = []
            lanes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
            for lane in lanes:
                for region in range(1, 11):
                    for color in ['R', 'G', 'B']:
                        key = f"{lane}{region}-{color}"
                        feature_vector.append(features.get(key, 0))
            
            feature_data.append(feature_vector)
            targets.append(row[target_column])
    
    print(f"✅ Successfully extracted features from {len(feature_data)} samples")
    
    if len(feature_data) < 5:
        print("❌ Not enough samples for correlation analysis")
        print(f"   Need at least 5 samples, got {len(feature_data)}")
        return None
    
    # Calculate correlations
    X = np.array(feature_data)
    y = np.array(targets)
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Target values range: {y.min():.1f} - {y.max():.1f}")
    
    correlations = []
    feature_names = []
    lanes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    for i, lane in enumerate(lanes):
        for region in range(1, 11):
            for color in ['R', 'G', 'B']:
                feature_name = f"{lane}{region}-{color}"
                feature_idx = i * 30 + (region-1) * 3 + ['R', 'G', 'B'].index(color)
                
                if feature_idx < X.shape[1]:
                    # Check if feature has variance
                    feature_values = X[:, feature_idx]
                    if np.std(feature_values) > 0:  # Only calculate correlation if there's variance
                        corr = np.corrcoef(feature_values, y)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                            feature_names.append(feature_name)
    
    print(f"📊 Valid correlations calculated: {len(correlations)}")
    
    if len(correlations) == 0:
        print("❌ No valid correlations calculated!")
        print("   Possible issues:")
        print("   • Features have no variance (all same values)")
        print("   • Target variable has no variance")
        print("   • Insufficient data diversity")
        return None
    
    # Visualize correlations - only if we have valid data
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation distribution
    axes[0, 0].hist(correlations, bins=min(30, len(correlations)//2), 
                    color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Feature Correlations')
    axes[0, 0].set_xlabel(f'Correlation with {target_column}')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Top positive correlations
    if len(correlations) >= 10:
        top_pos_idx = np.argsort(correlations)[-15:]
        top_pos_corr = [correlations[i] for i in top_pos_idx]
        top_pos_names = [feature_names[i] for i in top_pos_idx]
        
        axes[0, 1].barh(range(len(top_pos_corr)), top_pos_corr, 
                        color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Top Positive Correlations')
        axes[0, 1].set_xlabel(f'Correlation with {target_column}')
        axes[0, 1].set_yticks(range(len(top_pos_corr)))
        axes[0, 1].set_yticklabels(top_pos_names, fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Not enough data\\nfor detailed analysis', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Top Positive Correlations (Insufficient Data)')
    
    # 3. Top negative correlations
    if len(correlations) >= 10:
        top_neg_idx = np.argsort(correlations)[:15]
        top_neg_corr = [correlations[i] for i in top_neg_idx]
        top_neg_names = [feature_names[i] for i in top_neg_idx]
        
        axes[1, 0].barh(range(len(top_neg_corr)), top_neg_corr, 
                        color='red', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Top Negative Correlations')
        axes[1, 0].set_xlabel(f'Correlation with {target_column}')
        axes[1, 0].set_yticks(range(len(top_neg_corr)))
        axes[1, 0].set_yticklabels(top_neg_names, fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Not enough data\\nfor detailed analysis', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Top Negative Correlations (Insufficient Data)')
    
    # 4. Correlation by lane
    lane_correlations = {}
    for lane in lanes:
        lane_corrs = [corr for i, corr in enumerate(correlations) 
                     if feature_names[i].startswith(lane)]
        if lane_corrs:
            lane_correlations[lane] = np.mean(np.abs(lane_corrs))
        else:
            lane_correlations[lane] = 0
    
    axes[1, 1].bar(lane_correlations.keys(), lane_correlations.values(), 
                   color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Average Absolute Correlation by Lane')
    axes[1, 1].set_xlabel('Lane')
    axes[1, 1].set_ylabel('Mean |Correlation|')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation summary
    print("\n📊 Correlation Analysis Summary:")
    print(f"   • Features analyzed: {len(correlations)}")
    
    if correlations:
        print(f"   • Max positive correlation: {max(correlations):.3f}")
        print(f"   • Max negative correlation: {min(correlations):.3f}")
        print(f"   • Mean absolute correlation: {np.mean(np.abs(correlations)):.3f}")
        print(f"   • Standard deviation: {np.std(correlations):.3f}")
    else:
        print("   • ❌ No valid correlations calculated")
        print("   • This might be due to insufficient data or feature extraction issues")
    
    return {
        'correlations': correlations,
        'feature_names': feature_names,
        'lane_correlations': lane_correlations,
        'feature_matrix': X,
        'targets': y
    }