clc,clf,clear,close all;

n_bits = 120;
n_patterns_vector = [12,24,48,70,100,120];
diagonal_weights_zero = true;
n_iterations_per_pattern = 1e5;
% 1e5 tar ca 12 minuter

updated_bit_changed = zeros(n_iterations_per_pattern,length(n_patterns_vector));

for n_patterns_i = 1:length(n_patterns_vector)
    n_patterns = n_patterns_vector(n_patterns_i);
    for iteration_j = 1:n_iterations_per_pattern
        network = DeterministicHopfieldNetwork();
        network.set_diagonal_weights_to_zero(diagonal_weights_zero);

        patterns = generate_n_patterns(n_bits,n_patterns);
        network.set_patterns(patterns);
        network.generate_weights;

        pattern_to_feed_index = randi(n_patterns,1);
        original_pattern = patterns(:,pattern_to_feed_index);
        bit_to_update_index = randi(n_bits,1);

        updated_pattern = network.update_bit_of_pattern(original_pattern,bit_to_update_index);

        original_bit = original_pattern(bit_to_update_index);
        updated_bit = updated_pattern(bit_to_update_index);

        updated_bit_changed(iteration_j,n_patterns_i) = original_bit ~= updated_bit;
    end
end

one_step_error_probability = sum(updated_bit_changed,1)/n_iterations_per_pattern;
fprintf('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n',one_step_error_probability);