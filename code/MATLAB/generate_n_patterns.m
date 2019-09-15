function patterns = generate_n_patterns(n_bits,n_patterns)
    patterns = randi(2,n_bits,n_patterns);
    patterns = patterns - 1;
    patterns = 2*patterns - 1;
end