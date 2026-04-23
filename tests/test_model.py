import torch
import timm

def test_model_creation():
    print("Testing model creation...")
    try:
        model = timm.create_model('xception', pretrained=True, num_classes=1)
        print("Model created successfully!")
        
        # Test forward pass with dummy input
        print("Testing forward pass...")
        dummy_input = torch.randn(1, 3, 299, 299)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        if output.shape == (1, 1):
            print("Forward pass successful. Output shape is correct.")
        else:
            print(f"Forward pass successful but output shape {output.shape} might be unexpected (expected (1,1)).")
            
    except Exception as e:
        print(f"Model creation failed: {e}")

if __name__ == "__main__":
    test_model_creation()
