import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import both model versions
import custom_model as cm
import custom_model_v2 as cm2

def setup_environment():
    """Setup environment variables and configurations"""
    load_dotenv()
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Create directories for saving models and plots
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def basic_example():
    """Example using the streamlined version (custom_model_v2)"""
    print("\n=== Running Basic Example (custom_model_v2) ===")
    
    # Fetch data
    data = cm2.fetch_construction_data(
        symbols=["BTC/USDT"],
        timeframe="1h",
        since_days=30
    )
    
    # Create dataset
    dataset = cm2.ConstructionDataset(data)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # Initialize and train model
    model = cm2.ConstructionLSTM()
    train_losses, val_losses = cm2.train_model(
        model, train_loader, test_loader,
        num_epochs=50
    )
    
    # Save model
    cm2.save_model(model, None, 50, val_losses[-1], 'models/basic_model.pth')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Basic Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/basic_training.png')
    plt.close()

def advanced_example():
    """Example using the advanced version (custom_model)"""
    print("\n=== Running Advanced Example (custom_model) ===")
    
    # Fetch data with multiple symbols
    data = cm.fetch_construction_data(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        since_days=60
    )
    
    # Create dataset with enhanced features
    dataset = cm.ConstructionDataset(data)
    
    # Split data
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # Initialize enhanced model
    model = cm.ConstructionLSTM(
        input_size=17,
        hidden_size=128,
        num_layers=3,
        dropout=0.3
    )
    
    # Train model
    train_losses, val_losses = cm.train_model(
        model, train_loader, val_loader,
        num_epochs=100
    )
    
    # Evaluate model
    metrics = cm.evaluate_model_extended(model, test_loader, dataset.scaler)
    
    # Print metrics
    print("\nAdvanced Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate and plot predictions
    initial_sequence = dataset[0][0].unsqueeze(0)
    predictions = cm.multi_step_forecast(
        model, dataset, initial_sequence, steps=72
    )
    
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:, 3], label='Predicted Close Price')
    plt.title('72-Hour Price Prediction (Advanced Model)')
    plt.xlabel('Hours')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('plots/advanced_prediction.png')
    plt.close()

def compare_models():
    """Compare predictions from both models"""
    print("\n=== Comparing Models ===")
    
    # Fetch same data for both models
    data = cm.fetch_construction_data(
        symbols=["BTC/USDT"],
        timeframe="1h",
        since_days=30
    )
    
    # Create datasets
    basic_dataset = cm2.ConstructionDataset(data)
    advanced_dataset = cm.ConstructionDataset(data)
    
    # Load or train models
    basic_model = cm2.ConstructionLSTM()
    advanced_model = cm.ConstructionLSTM(input_size=17)
    
    # Generate predictions
    basic_seq = basic_dataset[0][0].unsqueeze(0)
    advanced_seq = advanced_dataset[0][0].unsqueeze(0)
    
    basic_pred = cm2.multi_step_forecast(
        basic_model, basic_dataset, basic_seq, steps=48
    )
    advanced_pred = cm.multi_step_forecast(
        advanced_model, advanced_dataset, advanced_seq, steps=48
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(basic_pred[:, 3], label='Basic Model')
    plt.plot(advanced_pred[:, 3], label='Advanced Model')
    plt.title('Model Comparison: 48-Hour Predictions')
    plt.xlabel('Hours')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('plots/model_comparison.png')
    plt.close()

def main():
    """Main execution function"""
    setup_environment()
    
    print("Cryptocurrency Trading System Examples")
    print("=====================================")
    
    # Run examples
    try:
        basic_example()
        advanced_example()
        compare_models()
    except Exception as e:
        print(f"Error during execution: {str(e)}")
    
    print("\nExecution completed. Check 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()