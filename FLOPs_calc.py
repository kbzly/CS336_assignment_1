def flops_calc(d_model, num_heads, d_ff, num_layers, context_length):
    flops = 0
    # 1. Multi-head self-attention
    flops_mha = 0 # 0 for now
    # 1.1. Q, K, V projection
    flops_mha += 2 * context_length * d_model * d_model # Q, K, V projection
    # 1.2. QK^T
    flops_mha += 2 * context_length * d_model * context_length # QK^T
    # 1.3. attention output
    flops_mha += 2 * context_length * context_length * d_model 
    # 1.4. output projection
    flops_mha += 2 * d_model * d_model * context_length

    # 2. Feed-forward network
    flops_ffn = 0 # 0 for now
    # 2.1. First linear layer
    flops_ffn += 2 * context_length * d_model * d_ff # First linear layer
    # 2.2. Second linear layer
    flops_ffn += 2 * context_length * d_ff * d_model # Second linear layer
    # 2.3. Third linear layer
    flops_ffn += 2 * context_length * d_model * d_ff # Third linear layer
    # 2.4. element-wise multiplication
    flops_ffn += 2 * context_length * d_ff
    # 2.5. W2 Linear layer
    flops_ffn += 2 * context_length * d_ff * d_model # W2 Linear layer
    
    return (flops_mha + flops_ffn) * num_layers

if __name__ == "__main__":
    # GPT-2 small
    d_model = 768
    num_heads = 12
    d_ff = 3072
    num_layers = 12
    context_length = 1024
    print(flops_calc(d_model, num_heads, d_ff, num_layers, context_length))
    
    # GPT-2 medium
    d_model = 1024
    num_heads = 16
    d_ff = 4096
    num_layers = 24
    context_length = 1024
    print(flops_calc(d_model, num_heads, d_ff, num_layers, context_length))
    
    # GPT-2 large
    d_model = 1280
    num_heads = 20
    d_ff = 5120
    num_layers = 36
    context_length = 1024
    print(flops_calc(d_model, num_heads, d_ff, num_layers, context_length))

    # GPT-2 XL 
    d_model = 1280
    num_heads = 20
    d_ff = 5120
    num_layers = 36
    context_length = 16384
    print(flops_calc(d_model, num_heads, d_ff, num_layers, context_length))

    # analysis
    

