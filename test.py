import torch
from mingpt.model import NewLayerNorm, NewGELU, NewLinear, activate_custom_layers

torch.manual_seed(42)

@torch.no_grad()
def test_custom_layernorm():
    hidden_size = 768
    input = torch.randn(15, hidden_size, dtype=torch.float32, device="cuda")
    layer = NewLayerNorm(hidden_size, device="cuda")
    weights = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    bias = 0.1 * torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    layer.weight = torch.nn.Parameter(weights)
    layer.bias = torch.nn.Parameter(bias)

    activate_custom_layers(False)
    reference = layer(input)

    activate_custom_layers(True)
    test = layer(input)

    passed = torch.allclose(reference, test, atol=1e-6)
    if passed:
        print(f"✅ Layernorm test passed!")
    else:
        print(f"❌ Layernorm test failed: {reference - test}")


@torch.no_grad()
def test_custom_gelu():
    hidden_size = 768
    input = torch.randn(15, hidden_size, dtype=torch.float32, device="cuda")
    layer = NewGELU()

    activate_custom_layers(False)
    reference = layer(input)

    activate_custom_layers(True)
    test = layer(input)

    passed = torch.allclose(reference, test)
    if passed:
        print(f"✅ GELU test passed!")
    else:
        print(f"❌ GELU test failed: {reference - test}")


@torch.no_grad()
def test_custom_linear():
    hidden_size = 768
    input = torch.randn(15, hidden_size, dtype=torch.float32, device="cuda")
    layer = NewLinear(hidden_size, 2 * hidden_size, device="cuda")
    weights = torch.zeros(2 * hidden_size, hidden_size, dtype=torch.float32, device="cuda")
    bias = 0.1 * torch.randn(2 * hidden_size, dtype=torch.float32, device="cuda")
    layer.weight = torch.nn.Parameter(weights)
    layer.bias = torch.nn.Parameter(bias)

    activate_custom_layers(False)
    reference = layer(input)

    activate_custom_layers(True)
    test = layer(input)

    passed = torch.allclose(reference, test, atol=1e-4)
    if passed:
        print(f"✅ Linear test passed!")
    else:
        print(f"❌ Linear test failed: {reference - test}")

if __name__ == "__main__":
    test_custom_gelu()
    test_custom_layernorm()
    test_custom_linear()
